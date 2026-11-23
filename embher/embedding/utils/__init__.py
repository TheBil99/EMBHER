"""
Utility modules.

- data: gene metadata, VCF samples, personal-genome dataset construction
- borzoi_utils: Borzoi/Flashzoi model loading, one-hot encoding, track lookup (needs torch)
- gpu: device selection (needs torch)

Names are resolved lazily (PEP 562), so importing this package, or `coordinates`,
does not import torch or pysam.
"""
import importlib

_EXPORTS = {
    "data": ["load_gene_metadata", "load_vcf_samples", "create_genome_dataset"],
    "borzoi_utils": [
        "BORZOI_SEQ_LEN", "BORZOI_BIN_SIZE", "BORZOI_NUM_BINS", "BORZOI_NUM_TRACKS",
        "BORZOI_PRED_LENGTH", "BORZOI_SUPPORTED_CONTEXT_WINDOWS", "BORZOI_MODEL_NAMES",
        "FLASHZOI_MODEL_NAMES", "DEFAULT_BORZOI_MODEL", "DEFAULT_FLASHZOI_MODEL",
        "get_model_name", "init_borzoi_model", "dna_to_onehot", "get_borzoi_track_indices",
    ],
    "gpu": ["get_best_gpu", "select_device", "get_device_info"],
}
_SOURCE = {name: mod for mod, names in _EXPORTS.items() for name in names}
__all__ = sorted(_SOURCE)


def __getattr__(name):
    if name in _SOURCE:
        return getattr(importlib.import_module(f".{_SOURCE[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
