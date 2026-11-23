"""Backend interface, registry and the Enformer adapter (tiny random model, CPU)."""
import numpy as np
import pytest

from embher.embedding.backends import (EmbeddingBackend, available_backends, embed_sequences,
                                       get_backend, register_backend)


class _Dummy(EmbeddingBackend):
    name, window_bp, default_layers = "dummy", 8, ("a", "b")

    def embed(self, sequence, layers=None):
        self.check_window(sequence)
        v = np.array([sequence.count(c) for c in "ACGT"], dtype=np.float32)
        return {l: v for l in (layers or self.default_layers)}


def test_registry_and_embed_sequences():
    register_backend("dummy", _Dummy)
    assert {"borzoi", "flashzoi", "enformer", "alphagenome", "dummy"} <= set(available_backends())
    b = get_backend("dummy")
    df = embed_sequences(b, ["AACCGGTT", "AAAAAAAA"], ids=["d1", "d2"], progress=False)
    assert df.shape == (2, 4) and list(df.index) == ["d1", "d2"] and df.loc["d2", 0] == 8
    with pytest.raises(ValueError):
        b.embed("ACGT")
    with pytest.raises(KeyError):
        get_backend("nope")


def test_enformer_backend_on_a_tiny_random_model():
    torch = pytest.importorskip("torch")
    enf = pytest.importorskip("enformer_pytorch")
    from embher.embedding.backends.enformer import ENFORMER_WINDOW_BP

    torch.manual_seed(0)
    model = enf.Enformer.from_hparams(dim=192, depth=1, heads=2, output_heads=dict(human=5),
                                      target_length=896, dim_divisible_by=8)
    b = get_backend("enformer", model=model, device="cpu")
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), ENFORMER_WINDOW_BP))
    out = b.embed(seq)
    v = out["embeddings/final"]
    assert v.ndim == 1 and v.shape[0] == 2 * 192 and np.isfinite(v).all()
    mid = ENFORMER_WINDOW_BP // 2                                   # inside the cropped centre
    seq2 = seq[:mid] + ("A" if seq[mid] != "A" else "C") + seq[mid + 1:]
    assert not np.allclose(v, b.embed(seq2)["embeddings/final"], rtol=0, atol=1e-9)   # a SNV moves it
    with pytest.raises(ValueError):
        b.embed(seq[:-1])
    with pytest.raises(ValueError):
        b.embed(seq, ["transformer/block_0"])
