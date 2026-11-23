"""
Captum driver for Borzoi targets.

`BorzoiAttributer` has the same shape as decima's `DecimaAttributer` (a model wrapped by a
scalar transform, a Captum method, `attribute(inputs) -> (B, 4, L)`), but wraps any
embher.embedding.attribution.targets module instead of Decima's own heads.

Methods (all signed, none take abs):
  inputxgradient      the default; for a one-hot input this is the gradient at the observed
                      base, one value per position once summed over the 4 channels
  saliency            raw gradient over all 4 bases; `correct_grad` removes the per-position
                      mean over bases (Majdandzic et al.), as gReLU / Decima do
  integratedgradients path method from a baseline (zeros, or dinucleotide shuffles averaged
                      over `n_shuffles`); the only one informative for MSEReadout on the
                      reference itself

Padding (all-zero one-hot columns, i.e. N or off-chromosome) is excluded by default:
attributions there are zeroed, since saliency and IG can be non-zero at a base that is not
in the input.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

METHODS = ("inputxgradient", "saliency", "integratedgradients")


def _is_mse_readout(target: nn.Module) -> bool:
    from embher.embedding.attribution.targets import MSEReadout

    return isinstance(getattr(target, "readout", None), MSEReadout)


class BorzoiAttributer:
    def __init__(self, target: nn.Module, method: str = "inputxgradient"):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.target = target.eval()
        self.method = method

    def _explainer(self):
        from captum.attr import InputXGradient, IntegratedGradients, Saliency

        return {"inputxgradient": InputXGradient, "saliency": Saliency,
                "integratedgradients": IntegratedGradients}[self.method](self.target)

    def attribute(self, x, mask_padding: bool = True, correct_grad: bool = False,
                  n_steps: int = 50, n_shuffles: int = 0, seed: int = 0) -> torch.Tensor:
        """Attribution of the target scalar to `x`; returns CPU tensors.

        `x` is either one window (B, 4, L) -> (B, 4, L), or a TUPLE of windows (e.g. a donor's two
        haplotypes, each (1, 4, L)) that the target consumes jointly as `target(*x)` -> one scalar;
        the result is then (len(x), 4, L), one attribution per input, from the one joint scalar.
        """
        if _is_mse_readout(self.target) and self.method != "integratedgradients":
            raise ValueError(
                "the MSE readout is zero, with zero gradient, at the reference sequence, so "
                f"'{self.method}' would return all zeros; use method='integratedgradients' "
                "with a baseline (n_shuffles > 0), or attribute a non-reference input")

        multi = isinstance(x, (tuple, list))
        xs = tuple(t.detach().float() for t in x) if multi else (x.detach().float(),)
        valid = tuple(t.sum(dim=1, keepdim=True) > 0 for t in xs)    # (B, 1, L) each
        explainer = self._explainer()
        inputs = xs if multi else xs[0]

        if self.method == "saliency":
            attr = explainer.attribute(inputs, abs=False)
        elif self.method == "inputxgradient":
            attr = explainer.attribute(inputs)
        else:
            if multi:
                raise NotImplementedError("integratedgradients over several inputs")
            attr = self._integrated_gradients(explainer, xs[0], valid[0], n_steps, n_shuffles, seed)

        attrs = tuple(attr) if multi else (attr,)
        out = []
        for a, v in zip(attrs, valid):
            if correct_grad:
                if self.method != "saliency":
                    raise ValueError("correct_grad only applies to method='saliency'")
                a = a - a.mean(dim=1, keepdim=True)
            if mask_padding:
                a = a * v
            out.append(a.detach().cpu())
        return torch.cat(out, dim=0) if multi else out[0]

    @staticmethod
    def _integrated_gradients(explainer, x, valid, n_steps, n_shuffles, seed):
        if n_shuffles <= 0:
            return explainer.attribute(x, n_steps=n_steps, internal_batch_size=1)
        if not bool(valid.all()):
            raise ValueError("dinucleotide-shuffled baselines need a window without padding")
        from tangermeme.ersatz import dinucleotide_shuffle

        # (B, n_shuffles, 4, L) on x's device; each shuffle is one IG path, then averaged
        baselines = dinucleotide_shuffle(x.cpu(), n=n_shuffles,
                                         random_state=seed).to(x.device).float()
        total = torch.zeros_like(x)
        for j in range(n_shuffles):
            total += explainer.attribute(x, baselines=baselines[:, j], n_steps=n_steps,
                                         internal_batch_size=1)
        return total / n_shuffles
