"""Tests for embher.embedding.attribution (attributer, readouts, io, donors, motif enrichment) on
synthetic data.

The real-model pieces (early-exit forward, ExpressionTarget) need Borzoi weights and a GPU, so
they are not covered here.
"""
import numpy as np
import pandas as pd
import pytest


# ---- attributer (torch) ------------------------------------------------------------
torch = pytest.importorskip("torch")
pytest.importorskip("captum")


class _Toy(torch.nn.Module):
    """s(x) = sum_i w[i] * x[base_i, i] + 0.5 * (sum x)^2 / L : known gradient."""

    def __init__(self, L=64):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, L), requires_grad=False)

    def forward(self, x):
        return (self.w * x).sum(dim=(1, 2)).unsqueeze(-1)


def _onehot(L=64, pad=0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((1, 4, L), np.float32)
    x[0, rng.integers(0, 4, L), np.arange(L)] = 1
    if pad:
        x[0, :, :pad] = 0
    return torch.from_numpy(x)


def test_inputxgradient_is_weight_at_observed_base_and_padding_masked():
    from embher.embedding.attribution.attributer import BorzoiAttributer

    toy, x = _Toy(), _onehot(pad=8)
    a = BorzoiAttributer(toy, "inputxgradient").attribute(x)
    np.testing.assert_allclose(a.numpy(), (toy.w * x).numpy(), atol=1e-6)
    assert (a[:, :, :8] == 0).all()


def test_saliency_correct_grad_centres_over_bases():
    from embher.embedding.attribution.attributer import BorzoiAttributer

    toy, x = _Toy(), _onehot()
    raw = BorzoiAttributer(toy, "saliency").attribute(x)
    np.testing.assert_allclose(raw.numpy(), toy.w[None].numpy(), atol=1e-6)
    cen = BorzoiAttributer(toy, "saliency").attribute(x, correct_grad=True)
    np.testing.assert_allclose(cen.mean(1).numpy(), 0.0, atol=1e-6)
    with pytest.raises(ValueError):
        BorzoiAttributer(toy, "inputxgradient").attribute(x, correct_grad=True)


def test_integrated_gradients_completeness_against_zero_baseline():
    from embher.embedding.attribution.attributer import BorzoiAttributer

    toy, x = _Toy(), _onehot()
    a = BorzoiAttributer(toy, "integratedgradients").attribute(x, n_steps=20)
    np.testing.assert_allclose(a.sum().item(), toy(x).item(), atol=1e-4)


def test_mse_readout_on_reference_is_refused_for_gradient_methods():
    from embher.embedding.attribution.attributer import BorzoiAttributer
    from embher.embedding.attribution.targets import MSEReadout

    class _Emb(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = MSEReadout(np.zeros(3))

        def forward(self, x):
            return x.sum(-1)[:, :3].sum(-1, keepdim=True)

    for method in ("inputxgradient", "saliency"):
        with pytest.raises(ValueError, match="zero gradient"):
            BorzoiAttributer(_Emb(), method).attribute(_onehot())


def test_readouts():
    from embher.embedding.attribution.targets import LinearReadout, MSEReadout, NormReadout

    e = torch.tensor([[3.0, 4.0]])
    assert NormReadout()(e).item() == pytest.approx(5.0)
    assert MSEReadout(np.array([1.0, 0.0]))(e).item() == pytest.approx((4 + 16) / 2)
    assert LinearReadout(np.array([1.0, 2.0]), np.array([1.0, 1.0]))(e).item() == pytest.approx(8.0)


def test_onehot_indices_mark_padding():
    from embher.embedding.attribution.io import onehot_to_indices

    x = _onehot(L=10, pad=3)[0].numpy()
    idx = onehot_to_indices(x)
    assert (idx[:3] == 4).all() and (idx[3:] < 4).all()


# ---- donor windows / joint haplotype targets --------------------------------------------
def test_to_reference_frame_identity_deletion_and_insertion():
    from embher.embedding.attribution.donors import to_reference_frame
    from embher.embedding.utils.coordinates import CoordinateTransformer

    attr = np.arange(2 * 20, dtype=float).reshape(2, 20)
    assert np.array_equal(to_reference_frame(attr, CoordinateTransformer([]), 100), attr)

    # 3-bp deletion at reference position 108: later reference positions sit 3 earlier in the haplotype
    out = to_reference_frame(attr, CoordinateTransformer([(0, 0), (108, -3)]), 100)
    assert np.array_equal(out[:, :8], attr[:, :8])
    assert np.array_equal(out[:, 8:], attr[:, 5:17])

    # 3-bp insertion: later reference positions sit 3 later; the last ones fall off the window -> 0
    out = to_reference_frame(attr, CoordinateTransformer([(0, 0), (108, 3)]), 100)
    assert np.array_equal(out[:, :8], attr[:, :8])
    assert np.array_equal(out[:, 8:17], attr[:, 11:20])
    assert (out[:, 17:] == 0).all()


def test_exon_bin_weights_are_fractional_overlaps_on_the_32bp_grid():
    from embher.embedding.attribution.donor_targets import exon_bin_weights
    from embher.embedding.attribution.donors import output_start

    win_start = 1000
    first_bin = output_start(win_start)
    # an exon covering the second half of bin 0 and the first half of bin 1
    b0, b1, w = exon_bin_weights([(first_bin + 16, first_bin + 48)], win_start)
    assert (b0, b1) == (0, 2)
    np.testing.assert_allclose(w, [0.5, 0.5])
    # a haplotype whose exon was shifted one whole bin later by an upstream insertion
    b0, b1, w = exon_bin_weights([(first_bin + 32 + 16, first_bin + 32 + 48)], win_start)
    assert (b0, b1) == (1, 3)


def test_joint_scalar_over_two_haplotypes_gets_one_attribution_each():
    from embher.embedding.attribution.attributer import BorzoiAttributer

    L = 32

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Parameter(torch.randn(4, L), requires_grad=False)
            self.w2 = torch.nn.Parameter(torch.randn(4, L), requires_grad=False)

        def forward(self, x1, x2):
            return ((self.w1 * x1).sum() + 2 * (self.w2 * x2).sum()).reshape(1, 1)

    toy, x1, x2 = Joint(), _onehot(L, seed=1), _onehot(L, seed=2)
    a = BorzoiAttributer(toy, "inputxgradient").attribute((x1, x2))
    assert a.shape == (2, 4, L)
    np.testing.assert_allclose(a[0].numpy(), (toy.w1 * x1[0]).numpy(), atol=1e-6)
    np.testing.assert_allclose(a[1].numpy(), (2 * toy.w2 * x2[0]).numpy(), atol=1e-6)
    s = BorzoiAttributer(toy, "saliency").attribute((x1, x2))
    np.testing.assert_allclose(s[1].numpy(), (2 * toy.w2).numpy(), atol=1e-6)
    with pytest.raises(NotImplementedError):
        BorzoiAttributer(toy, "integratedgradients").attribute((x1, x2))


def test_motif_enrichment_flags_overrepresented_tf_and_not_null_like_ones():
    from embher.embedding.attribution import motif_enrichment as M

    assert M.tf_name("SPI1.H13CORE.0.P.B") == "SPI1"
    # 100 seqlet windows; SPI1 in 30 of them against a null rate of 5%, CTCF at the null rate
    hits = pd.DataFrame({"sequence": [str(i) for i in range(30)] + [str(i) for i in range(5)],
                         "motif": ["SPI1.H13CORE.0.P.B"] * 30 + ["CTCF.H13CORE.0.P.B"] * 5})
    obs = M.window_tf_counts(hits, 100)
    assert obs["SPI1"] == 30 and obs["CTCF"] == 5
    null = pd.Series({"SPI1": 500.0, "CTCF": 500.0})              # 5% of 10,000 null windows
    res = M.enrichment(obs, 100, null, 10000).set_index("tf")
    assert bool(res.loc["SPI1", "enriched"]) and not bool(res.loc["CTCF", "enriched"])
    assert res.loc["SPI1", "fold"] == pytest.approx(30 / 5, rel=0.05)


def test_motif_enrichment_min_support_and_empty_inputs():
    from embher.embedding.attribution import motif_enrichment as M

    obs = pd.Series({"A": 1.0})                                   # below min_obs=2: untested
    assert len(M.enrichment(obs, 50, pd.Series(dtype=float), 1000)) == 0
    assert len(M.enrichment(pd.Series(dtype=float), 0, pd.Series(dtype=float), 1000)) == 0
    q = M.bh(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(q >= np.array([0.01, 0.04, 0.03, 0.5])) and q.max() <= 1.0
    assert q[0] == pytest.approx(0.04)                            # 0.01*4/1, monotone-capped at 0.04
