#!/usr/bin/env python3
"""
Check that the hooked Borzoi capture reproduces a plain forward pass, and that
the captured tensors' declared geometry matches the model's actual layout.

PyTorch forward hooks that return None cannot alter outputs, so the real risks
here are (a) the post-crop coordinate mapping and (b) reading the wrong module.
This compares captured `predictions/human` against a plain model(x) call and
re-derives every tensor's resolution from its row count.

Usage:
    python tmp/scripts/validate_borzoi.py --weights_path $BORZOI_WEIGHTS
"""

import argparse
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights_path', default=os.environ.get('BORZOI_WEIGHTS'))
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    from embher.embedding.utils import init_borzoi_model, BORZOI_SEQ_LEN
    from embher.embedding.utils.borzoi_utils import dna_to_onehot
    from embher.embedding.capture import borzoi as bzcap

    model, tracks_df = init_borzoi_model(model_name=args.weights_path, device='auto')
    device = next(model.parameters()).device

    rng = np.random.default_rng(args.seed)
    sequence = ''.join(rng.choice(list('ACGT'), size=BORZOI_SEQ_LEN))

    # Capture over the whole window so nothing is cropped away.
    captured = bzcap.run_forward(
        model, sequence,
        window_bp=BORZOI_SEQ_LEN, win_start=0,
        crop_start=0, crop_end=BORZOI_SEQ_LEN,
    )

    # Plain forwards, no hooks. run_forward uses bf16 autocast on CUDA, so
    # compare against both that and full fp32: the bf16 row isolates the hooks,
    # the fp32 row shows what the autocast itself costs against stock Borzoi.
    def plain(autocast: bool):
        with torch.no_grad():
            x = dna_to_onehot(sequence, BORZOI_SEQ_LEN).to(device)
            if autocast and device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x, is_human=True)
            else:
                out = model(x, is_human=True)
        out = out[0].detach().float().cpu().numpy()
        return out.T if out.shape[0] != captured['predictions/human'][0].shape[0] else out

    got = captured['predictions/human'][0].astype(np.float32)
    for label, autocast in (('bf16 autocast (same as capture)', True),
                            ('fp32 (stock Borzoi)', False)):
        ref = plain(autocast)
        if got.shape != ref.shape:
            print(f"predictions/human vs {label}: SHAPE MISMATCH {got.shape} {ref.shape}")
            continue
        delta = float(np.abs(got - ref).max())
        scale = float(np.abs(ref).max()) or 1.0
        print(f"predictions/human vs {label:34s} max|Δ|={delta:10.6g} "
              f"rel={delta/scale:9.3g}")

    print(f"\n{'tensor':30s} {'shape':18s} {'res(attr)':>9s} {'res(derived)':>13s}  status")
    bad = 0
    for name, (arr, attrs) in sorted(captured.items()):
        res = int(attrs['resolution'])
        span = int(attrs['genomic_end']) - int(attrs['genomic_start'])
        derived = span // arr.shape[0]
        ok = derived == res and span == arr.shape[0] * res
        bad += not ok
        print(f"{name:30s} {str(arr.shape):18s} {res:9d} {derived:13d}  "
              f"{'OK' if ok else 'MISMATCH'}")

    print(f"\n{len(captured)} tensors, {bad} geometry mismatches")
    print("PASS" if bad == 0 else "FAIL")


if __name__ == '__main__':
    main()
