"""
Embedding tools: per-layer embedding capture and attribution for sequence-to-function
models (Borzoi/Flashzoi, AlphaGenome) on personal genomes, and embedding whitening.

Derived from alphagenome-personal-eval (Tu et al., Mostafavi lab); see the README.

- whiten: top-k whitened PC scores of an embedding (numpy only)
- dataset: personal genomes (GenomeDataset; needs torch, pysam)
- capture: capture intermediate activations to HDF5
- attribution: layer-wise attribution of Borzoi windows
- cli: command-line tools (see the console scripts in pyproject.toml)

Nothing is imported here, so the torch-free parts stay import-light.
"""
