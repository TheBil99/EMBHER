<p align="center"><img src="docs/logo.png" width="220" alt="embher"></p>

# embher

Two parts, installable separately:

- **`embher.heritability`**: heritability estimation from kernels (GRMs): exact multi-component AI-REML and Haseman-Elston regression, plus the helpers to build the kernels.
- **`embher.embedding`**: per-layer embedding capture and attribution for sequence-to-function models (Borzoi/Flashzoi, AlphaGenome for capture) on personal genomes, and embedding whitening.

## Install

```bash
pip install "embher @ git+https://github.com/TheBil99/EMBHER.git"                         # heritability + embedding tools
pip install "embher[capture-borzoi] @ git+https://github.com/TheBil99/EMBHER.git"         # + Borzoi/Flashzoi capture
pip install "embher[capture-enformer] @ git+https://github.com/TheBil99/EMBHER.git"        # + Enformer embeddings
pip install "embher[attribution] @ git+https://github.com/TheBil99/EMBHER.git"            # + attribution (implies capture-borzoi)
pip install "embher[capture-alphagenome] @ git+https://github.com/TheBil99/EMBHER.git"    # + AlphaGenome capture
```

Append `@<tag>` or `@<commit>` (full SHA) to the URL to pin a release. Extras combine, e.g. `embher[capture-borzoi,attribution]`. Every dependency of the base install, `capture-borzoi`, `capture-enformer` and `attribution` is on PyPI; `capture-alphagenome` also needs a manual step (below).

| Install | Adds |
|---|---|
| `embher` | numpy, pandas, scipy, pyarrow, pysam, pyyaml, tqdm, h5py, ml_dtypes, numba |
| `embher[capture-borzoi]` | torch, borzoi-pytorch |
| `embher[capture-enformer]` | torch, enformer-pytorch |
| `embher[capture-alphagenome]` | alphagenome (also jax, haiku, orbax, jmp and [alphagenome_research](https://github.com/google-deepmind/alphagenome_research), not on PyPI) |
| `embher[attribution]` | `capture-borzoi` + captum, tangermeme, decima, pyBigWig |
| `embher[all]` | `capture-borzoi`, `capture-enformer`, `attribution` and `dev` (not AlphaGenome) |
| `embher[dev]` | pytest, pytest-cov |

**AlphaGenome.** `alphagenome_research` is not on PyPI. After `pip install "embher[capture-alphagenome] @ git+https://github.com/TheBil99/EMBHER.git"`, install it from GitHub in the same environment, together with the jax stack it needs:

```bash
pip install alphagenome==0.9.0 \
    "alphagenome_research @ git+https://github.com/google-deepmind/alphagenome_research.git@1e55dcffb98ba26b31e74edc5e9f038f54c0e89d" \
    "jax[cuda12]==0.7.1" dm-haiku==0.0.16 orbax-checkpoint==0.11.33 jmp==0.0.4
```

These are the versions of the environment the AlphaGenome capture was run in (Python 3.12). Use `jax==0.7.1` instead of `jax[cuda12]==0.7.1` on a machine without a GPU. Other `alphagenome_research` versions expose the same classes and functions the capture code uses (checked from `v0.1.0` to current `main`), but only this combination has been run. Python >= 3.10. The Borzoi and AlphaGenome backends need disjoint stacks (torch versus jax): keep them in separate environments. Weights are not included: Borzoi/Flashzoi come from HuggingFace unless a local path is given; AlphaGenome needs a local checkpoint.

## Example

![embher schematic](docs/schematic.png)

[`examples/heritability_and_embedding_demo.ipynb`](examples/heritability_and_embedding_demo.ipynb) runs on simulated data with the base install, no weights and no GPU: genotype heritability by REML and HE, then an embedding whitened and used as a kernel. A flag swaps the simulated embedding for Borzoi's (needs `capture-borzoi` and a GPU).

## Modules

| Module | Contents |
|---|---|
| `embher.heritability.reml` | genotype/kernel GRMs (`build_grm`, `bin_masks`), covariate design (`design_matrix`), exact AI-REML (`fit_reml`, optional `nonneg=True` with LRT) |
| `embher.heritability.he` | Haseman-Elston regression with delete-one jackknife (`he_fit`) |
| `embher.heritability.estimate` | `estimate_h2`: REML or HE for several phenotypes on a list of kernels (behind `embher-h2`) |
| `embher.heritability.io` | `read_bed`: PLINK 1 `.bed` to donor x variant 0/1/2 dosage |
| `embher.heritability.grm_io` | `write_gcta_grm` |
| `embher.embedding.whiten` | `whiten_embeddings`: top-k whitened PC scores of an embedding; `embedding_kernel`: its kernel (numpy only) |
| `embher.embedding.backends` | `get_backend`, `embed_sequences`: one interface over Borzoi, Flashzoi, Enformer, AlphaGenome |
| `embher.embedding.dataset` | `GenomeDataset`: personal genomes (needs torch, pysam) |
| `embher.embedding.capture` | capture intermediate activations to HDF5 |
| `embher.embedding.attribution` | layer-wise attribution of Borzoi windows |
| `embher.embedding.utils` | gene metadata, VCF samples, coordinates, Borzoi helpers, device selection |

The most used `heritability` functions are re-exported: `from embher.heritability import fit_reml, he_fit, build_grm, read_bed`. The base install has no torch: `heritability`, `embedding.whiten` and `embedding.utils.coordinates` import without it.

## Command-line tools

| Command | Needs | Purpose |
|---|---|---|
| `embher-h2` | base install | heritability of phenotypes on genotype and/or embedding kernels (REML or HE) |
| `embher-kernel` | base install | build a kernel (`.npz`, optionally GCTA format) from an embedding or PLINK genotypes |
| `embher-capture` | `capture-borzoi` or `capture-alphagenome` | write every intermediate and final tensor of a forward pass to HDF5 |
| `embher-attribute` | `attribution` | attribute the L2 norm of each captured layer back to the input bases (Captum input x gradient) |
| `embher-inspect-h5` | base install | check a capture file |
| `embher-prepare-gene-meta` | base install | build the gene table `embher-capture` needs from a GENCODE parquet |

### Heritability

```bash
# genotype heritability, 4 MAF bins, with covariates
embher-h2 --pheno expression.tsv --pheno-col GENE1 --covar covariates.tsv \
    --bfile region --maf-bins 0,0.01,0.05,0.1,0.5 --maf-file region.maf --out h2.tsv

# genotype + embedding fitted jointly on the same donors (HE instead of REML)
embher-h2 --pheno expression.tsv --bfile region --embeddings embeddings.parquet \
    --estimator he --out h2_joint.tsv

# embedding kernel once, reused
embher-kernel --embeddings embeddings.parquet --out gene1_kernel      # whitened top 70% of the rank
embher-h2 --pheno expression.tsv --kernel gene1_kernel.npz --nonneg --out h2.tsv
```

Phenotype, embedding and covariate tables are donor-indexed (first column = donor id; parquet, tsv or csv). Kernels are built over the donors that have a phenotype value and appear in every input. The output has one row per phenotype: `h2`, `h2_se`, convergence flags, `n_samples`, and with several kernels `h2_genotype`, `h2_embedding`, ... `--nonneg` constrains the variance components to be non-negative and adds `lrt`, `lrt_p`.

### Capture and attribution

```bash
embher-capture --model_type borzoi \
    --gene_meta_path genes.tsv --vcf_file_path samples.vcf.gz \
    --hg38_file_path hg38.fa --gencode_parquet gencode.annotation.gtf.parquet \
    --n_genes 5 --n_samples 10 --output_h5 out/borzoi_tensors.h5
```

`genes.tsv` needs `gene_id, chr_hg38, tss_hg38, start_hg38, end_hg38`. Arguments can also come from a YAML `--config` (with `${ENV_VAR}` expansion); CLI arguments override it. See `embher-capture --help` for reductions (pooling, output aggregation over gene/exon/intron), the HDF5 layout, `--resume` and `--embedding_layers`. `embher.embedding.cli.validate_borzoi` and `validate_alphagenome` (run with `python -m`) check that the hooked capture reproduces the model's own forward pass.

## Model backends

`embher.embedding.backends` puts Borzoi, Flashzoi, Enformer and AlphaGenome behind one interface: a window of DNA in, mean-pooled embeddings per layer out. Swapping the model is a one-word change.

```python
from embher.embedding.backends import get_backend, embed_sequences
from embher.embedding.whiten import embedding_kernel

backend = get_backend("enformer")                       # "borzoi", "flashzoi", "alphagenome"
Z = embed_sequences(backend, windows, ids=donors)       # donors x features, one layer
K, k = embedding_kernel(Z.to_numpy())                   # whitened kernel, ready for fit_reml
```

Windows must be exactly `backend.window_bp` long (Borzoi 524,288 bp, Enformer 196,608 bp, AlphaGenome 2 kb to 1 Mb). To add a model, subclass `EmbeddingBackend` (set `name`, `window_bp`, `default_layers`, implement `embed`) and call `register_backend`. The Borzoi and Enformer backends are tested with their pretrained weights; the AlphaGenome backend is not covered by the test suite.

## Acknowledgements

The framework for loading personal genomes (`GenomeDataset`, `CoordinateTransformer` and the Borzoi helpers) comes from [alphagenome-personal-eval](https://github.com/mostafavilabuw/alphagenome-personal-eval) (Tu et al., Mostafavi lab). Adapted code says so in its docstring. Please cite their work if you use it:

> Xinming Tu, Anna Spiro, Maria Chikina, Sara Mostafavi. *A modality gap in personal-genome prediction by sequence-to-function model.* bioRxiv (2026). [Preprint](https://www.biorxiv.org/content/10.64898/2026.02.01.702969v1)

## Citation

If you use EMBHER, please cite:

> Federico Billeci, Miquel Anglada-Girotto, Alexander Gusev, Jonathan Frazer & Mafalda Dias. *Sequence-to-function embeddings across individuals capture rare and common variant heritability.* bioRxiv (forthcoming; DOI to be added).

The full analysis pipeline of the manuscript is at [TheBil99/EMBHER_publication](https://github.com/TheBil99/EMBHER_publication).
