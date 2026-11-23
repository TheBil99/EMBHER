"""embher-h2 and embher-kernel on a small simulated cohort (numpy only, no torch)."""
import numpy as np
import pandas as pd
import pytest

from embher.cli import h2 as h2_cli
from embher.cli import kernel as kernel_cli
from embher.heritability import build_grm, fit_reml, read_bed

N, M = 500, 800


def _write_bed(prefix, dosage):
    """dosage (n x m of 0/1/2) -> SNP-major PLINK 1 files (bim's second allele is counted)."""
    code = np.array([0, 2, 3])[dosage.astype(int)]                 # 0 -> hom A1, 1 -> het, 2 -> hom A2
    n = dosage.shape[0]
    pad = (-n) % 4
    bits = np.pad(code.T, ((0, 0), (0, pad)))                      # m x (n+pad)
    b = bits.reshape(dosage.shape[1], -1, 4)
    packed = (b[..., 0] | (b[..., 1] << 2) | (b[..., 2] << 4) | (b[..., 3] << 6)).astype(np.uint8)
    with open(f"{prefix}.bed", "wb") as fh:
        fh.write(bytes([0x6C, 0x1B, 0x01]))
        fh.write(packed.tobytes())
    pd.DataFrame({"f": [f"d{i}" for i in range(n)], "i": [f"d{i}" for i in range(n)],
                  "p": 0, "m": 0, "s": 0, "ph": -9}).to_csv(f"{prefix}.fam", sep=" ", header=False, index=False)
    pd.DataFrame({"c": 1, "id": [f"v{j}" for j in range(dosage.shape[1])], "cm": 0,
                  "bp": np.arange(dosage.shape[1]), "a1": "A", "a2": "G"}).to_csv(
        f"{prefix}.bim", sep="\t", header=False, index=False)


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    d = tmp_path_factory.mktemp("cohort")
    rng = np.random.default_rng(1)
    maf = rng.uniform(0.05, 0.5, M)
    dosage = rng.binomial(2, maf, (N, M))
    z = (dosage - 2 * maf) / np.sqrt(2 * maf * (1 - maf))
    g = z @ rng.normal(0, np.sqrt(0.5 / M), M)
    y = g + rng.normal(0, np.sqrt(0.5), N)
    ids = [f"d{i}" for i in range(N)]
    Z = 1.5 * np.outer(g / g.std(), rng.normal(size=60)) + rng.normal(size=(N, 60))
    _write_bed(str(d / "geno"), dosage)
    pd.DataFrame({"IID": ids, "expr": y}).to_csv(d / "pheno.tsv", sep="\t", index=False)
    pd.DataFrame(Z, index=pd.Index(ids, name="IID")).to_parquet(d / "emb.parquet")
    return d, dosage, y


def test_read_bed_roundtrip(cohort):
    d, dosage, _ = cohort
    _, back = read_bed(str(d / "geno"))
    assert np.array_equal(back, dosage)


def test_h2_genotype_reml_matches_library(cohort):
    d, dosage, y = cohort
    out = d / "g.tsv"
    h2_cli.main(["--pheno", str(d / "pheno.tsv"), "--bfile", str(d / "geno"), "--out", str(out)])
    res = pd.read_csv(out, sep="\t")
    K, _ = build_grm(dosage.astype(np.float32))
    direct = fit_reml([K], y, np.ones((N, 1)))
    assert res.loc[0, "h2"] == pytest.approx(direct["h2"], abs=1e-6)
    assert res.loc[0, "n_samples"] == N and bool(res.loc[0, "converged"])
    assert 0.2 < res.loc[0, "h2"] < 0.8


def test_h2_joint_and_he(cohort):
    d, _, _ = cohort
    out = d / "j.tsv"
    h2_cli.main(["--pheno", str(d / "pheno.tsv"), "--bfile", str(d / "geno"),
                 "--embeddings", str(d / "emb.parquet"), "--out", str(out)])
    res = pd.read_csv(out, sep="\t")
    assert {"h2_genotype", "h2_embedding", "h2_genotype_se"} <= set(res.columns)
    out_he = d / "he.tsv"
    h2_cli.main(["--pheno", str(d / "pheno.tsv"), "--embeddings", str(d / "emb.parquet"),
                 "--estimator", "he", "--out", str(out_he)])
    he = pd.read_csv(out_he, sep="\t")
    assert he.loc[0, "estimator"] == "he" and np.isfinite(he.loc[0, "h2_se"])


def test_kernel_npz_equals_on_the_fly(cohort):
    d, _, _ = cohort
    kernel_cli.main(["--embeddings", str(d / "emb.parquet"), "--gcta", "--out", str(d / "kemb")])
    assert (d / "kemb.grm.bin").exists() and (d / "kemb.grm.id").exists()
    a, b = d / "a.tsv", d / "b.tsv"
    h2_cli.main(["--pheno", str(d / "pheno.tsv"), "--embeddings", str(d / "emb.parquet"), "--out", str(a)])
    h2_cli.main(["--pheno", str(d / "pheno.tsv"), "--kernel", str(d / "kemb.npz"), "--out", str(b)])
    ha, hb = pd.read_csv(a, sep="\t").h2[0], pd.read_csv(b, sep="\t").h2[0]
    assert ha == pytest.approx(hb, abs=1e-6)


def test_shuffled_embedding_has_no_signal(cohort):
    d, _, _ = cohort
    emb = pd.read_parquet(d / "emb.parquet")
    emb.index = emb.index.to_numpy()[np.random.default_rng(0).permutation(N)]
    emb.to_parquet(d / "emb_shuffled.parquet")
    out = d / "s.tsv"
    h2_cli.main(["--pheno", str(d / "pheno.tsv"), "--embeddings", str(d / "emb_shuffled.parquet"),
                 "--out", str(out)])
    assert pd.read_csv(out, sep="\t").h2[0] < 0.15
