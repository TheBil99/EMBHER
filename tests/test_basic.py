"""Import smoke tests."""


def test_heritability_and_whiten_are_torch_free():
    # fresh interpreter: other tests in this process import torch
    import subprocess, sys
    code = (
        "import sys\n"
        "from embher.heritability import fit_reml, he_fit, read_bed\n"
        "from embher.embedding.whiten import whiten_embeddings\n"
        "from embher.embedding.utils import coordinates\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_package():
    import embher
    assert hasattr(embher, "__version__")


def test_utils_exports_used_by_capture_forward():
    from embher.embedding.utils import (  # noqa: F401
        BORZOI_SEQ_LEN, create_genome_dataset, get_borzoi_track_indices,
        init_borzoi_model, load_gene_metadata, load_vcf_samples,
    )
    from embher.embedding.capture.regions import SEQUENCE_LENGTHS
    assert SEQUENCE_LENGTHS["1MB"] == 1048576


def test_capture_and_attribution_modules_import():
    from embher.embedding.capture import alphagenome, reduce, regions  # noqa: F401  (jax is imported lazily)
    from embher.embedding.capture.h5_writer import ForwardH5Writer, StackedH5Writer  # noqa: F401
    from embher.embedding.attribution import capture_layers, donors  # noqa: F401
    from embher.embedding import dataset as PersonalDataset  # noqa: F401
