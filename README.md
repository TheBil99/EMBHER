# embher

Heritability estimation from kernels (GRMs): exact multi-component AI-REML and Haseman-Elston regression, plus the helpers to build the kernels.

## Install

```bash
pip install git+ssh://git@github.com/TheBil99/EMBHER.git@v0.1.1
```

Requires Python >= 3.9, numpy, pandas, scipy.

## Modules

| Module | Contents |
|---|---|
| `embher.reml` | genotype/kernel GRMs (`build_grm`, `bin_masks`), covariate design (`design_matrix`), exact AI-REML (`fit_reml`, optional `nonneg=True` with LRT) |
| `embher.he` | Haseman-Elston regression with delete-one jackknife (`he_fit`) |
| `embher.io` | `read_bed`: PLINK 1 `.bed` to donor x variant 0/1/2 dosage |
| `embher.pca` | `whiten_embeddings`: top-k whitened PC scores of an embedding |
| `embher.grm_io` | `write_gcta_grm` |

## Paper

The analyses that reproduce the paper's results live in a separate analysis repository.
