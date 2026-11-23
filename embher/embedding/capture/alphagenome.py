"""
Local AlphaGenome model loading and instrumented forward passes.

Unlike alphagenome.models.dna_client (the hosted gRPC API), alphagenome_research
loads real JAX/Haiku checkpoint weights locally. That gives us access to every
module's forward output via hk.intercept_methods, which the hosted API can
never expose - it only ever returns final track predictions.

Captured per forward pass:
  encoder/bin_size_{1,2,4,8,16,32,64}, encoder/trunk
  transformer/block_{0..8}          (pair activations dropped)
  decoder/bin_size_{64,32,16,8,4,2,1}
  embeddings/{1bp,128bp}            (pair embeddings dropped)
  predictions/<head>/res_{1,128}    (coverage heads; only output_heads when
                                     aggregating, every coverage head raw when not)

All imports of jax/haiku/alphagenome_research are deferred into function
bodies, since these are heavy, rarely-installed dependencies (unlike the
hosted alphagenome package, they aren't declared anywhere in this project).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from embher.embedding.capture import tracks as track_meta
from embher.embedding.capture.crop import crop_bins
from embher.embedding.capture.reduce import OutputAggregation, Pooling, build_requests

# Coverage (genome-tracks) heads only - excludes contact_maps and the three
# splice-site heads, per the scope agreed for capture_forward.py.
COVERAGE_HEADS = (
    'atac', 'dnase', 'procap', 'cage', 'rna_seq', 'chip_tf', 'chip_histone'
)

# The embeddings the output heads read (both resolutions), the ones capture_forward.py's
# --embedding_layers final keeps.
FINAL_EMBEDDINGS = ('embeddings/1bp', 'embeddings/128bp')

# SequenceDecoder upsamples through these bin sizes in this exact order
# (see alphagenome_research.model.model.SequenceDecoder.__call__).
DECODER_BIN_SIZES = (64, 32, 16, 8, 4, 2, 1)


@dataclass
class LoadedModel:
    """A loaded checkpoint plus the instrumented forward pass that runs it."""

    params: Any
    state: Any
    metadata: Any
    one_hot_encoder: Any
    organism: Any
    device: Any
    forward_fn: Any


def load_model(
    weights_path: str,
    organism: Optional[str] = None,
    device: Optional[str] = None,
) -> LoadedModel:
    """
    Load an AlphaGenome checkpoint from local weights for instrumented forward passes.

    Mirrors alphagenome_research.model.dna_model.create()'s checkpoint loading,
    but does not construct an AlphaGenomeModel: that class refuses to build on
    anything but a GPU/TPU, and we only need params/state/metadata anyway.

    Args:
        weights_path: Path to the AlphaGenome checkpoint directory.
        organism: 'human' or 'mouse' (default: 'human').
        device: JAX device string, e.g. 'cpu', 'gpu:0'. Defaults to the first
            available local device.

    Returns:
        LoadedModel ready to pass to run_forward().
    """
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from alphagenome.models import dna_model as dna_model_types
    from alphagenome_research.model import dna_model as local_dna_model
    from alphagenome_research.model import one_hot_encoder as one_hot_encoder_mod
    from alphagenome_research.model.metadata import metadata as metadata_lib

    organism_map = {
        'human': dna_model_types.Organism.HOMO_SAPIENS,
        'mouse': dna_model_types.Organism.MUS_MUSCULUS,
    }
    organism_enum = organism_map[organism] if organism else dna_model_types.Organism.HOMO_SAPIENS

    if device is None:
        jax_device = jax.local_devices()[0]
    else:
        backend, _, idx = device.partition(':')
        devices = jax.devices(backend or None)
        jax_device = devices[int(idx)] if idx else devices[0]

    # AlphaGenome's checkpoint architecture bakes in slots for every organism
    # in default_organism_settings() (num_organisms=2), so metadata must cover
    # all of them even if we only ever run predictions for one.
    organism_settings = local_dna_model.default_organism_settings()
    metadata = {org: metadata_lib.load(org) for org in organism_settings}

    model_settings = local_dna_model.ModelSettings()
    init_fn, _, _, _, _ = local_dna_model.create_model(
        metadata,
        num_splice_sites=model_settings.num_splice_sites,
        splice_site_threshold=model_settings.splice_site_threshold,
    )

    dna_sequence_shape = jax.ShapeDtypeStruct((1, 2048, 4), dtype=jnp.float32)
    organism_index_shape = jax.ShapeDtypeStruct((1,), dtype=jnp.int32)
    target_shapes = jax.eval_shape(
        init_fn, jax.random.PRNGKey(0), dna_sequence_shape, organism_index_shape
    )
    checkpointer = ocp.StandardCheckpointer()
    params, state = checkpointer.restore(weights_path, target=target_shapes)

    return LoadedModel(
        params=jax.device_put(params, jax_device),
        state=jax.device_put(state, jax_device),
        metadata=metadata,
        one_hot_encoder=one_hot_encoder_mod.DNAOneHotEncoder(),
        organism=organism_enum,
        device=jax_device,
        forward_fn=_build_capturing_forward(
            metadata,
            model_settings.num_splice_sites,
            model_settings.splice_site_threshold,
        ),
    )


def _build_capturing_forward(metadata, num_splice_sites: int, splice_site_threshold: float):
    """Builds a jitted forward pass that returns every tensor we store, using
    hk.intercept_methods as the JAX/Haiku analogue of a PyTorch forward hook."""
    import haiku as hk
    import jax
    import jmp
    from alphagenome_research.model import attention, convolutions
    from alphagenome_research.model import model as ag_model

    # The same mixed-precision policy dna_model.create_model() applies to its
    # own forward pass. Without it the model runs in float32 compute, which is
    # both slower and numerically different from real AlphaGenome inference.
    jmp_policy = jmp.get_policy('params=float32,compute=bfloat16,output=bfloat16')

    def _make_interceptor(captured: Dict[str, Any], counters: Dict[str, int]):
        def interceptor(next_f, args, kwargs, context):
            out = next_f(*args, **kwargs)
            if context.method_name != '__call__':
                return out

            module = context.module
            if isinstance(module, ag_model.SequenceEncoder):
                trunk, intermediates = out
                for key, value in intermediates.items():
                    captured[f'encoder/{key}'] = value
                captured['encoder/trunk'] = trunk
            elif isinstance(module, convolutions.UpResBlock):
                i = counters['decoder']
                captured[f'decoder/bin_size_{DECODER_BIN_SIZES[i]}'] = out
                counters['decoder'] = i + 1
            elif isinstance(module, attention.MLPBlock):
                # x after the block is not itself a module output - it's the
                # residual `x += MLPBlock()(x)` in TransformerTower - so we
                # reconstruct it from the block's own input and output.
                i = counters['transformer']
                captured[f'transformer/block_{i}'] = args[0] + out
                counters['transformer'] = i + 1
            return out

        return interceptor

    @hk.transform_with_state
    def _forward(dna_sequence, organism_index):
        captured: Dict[str, Any] = {}
        counters = {'decoder': 0, 'transformer': 0}
        with hk.mixed_precision.push_policy(ag_model.AlphaGenome, jmp_policy), \
                hk.intercept_methods(_make_interceptor(captured, counters)):
            predictions, embeddings = ag_model.AlphaGenome(
                metadata,
                num_splice_sites=num_splice_sites,
                splice_site_threshold=splice_site_threshold,
            )(dna_sequence, organism_index)

        captured['embeddings/1bp'] = embeddings.embeddings_1bp
        captured['embeddings/128bp'] = embeddings.embeddings_128bp

        # Selecting heads here rather than after apply() lets XLA drop the ones
        # we never store (contact maps and the three splice heads) entirely.
        for head in COVERAGE_HEADS:
            head_predictions = predictions[head]
            for resolution in (1, 128):
                key = f'scaled_predictions_{resolution}bp'
                if key in head_predictions:
                    captured[f'predictions/{head}/res_{resolution}'] = head_predictions[key]
        return captured

    return jax.jit(_forward.apply)


def run_forward(
    model: LoadedModel,
    sequence: str,
    window_bp: int,
    win_start: int,
    crop_start: int,
    crop_end: int,
    embedding_crop: bool = True,
    poolings: Sequence[Pooling] = (),
    outputs: Optional[OutputAggregation] = None,
    output_heads: Sequence[str] = ('rna_seq',),
    embedding_layers: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Run one instrumented forward pass, reducing tensors before they leave the device.

    Pooling and aggregation run on device deliberately: an uncropped 1MB
    embeddings/1bp tensor is gigabytes, and only its reduction is ever kept.

    Args:
        model: Loaded checkpoint from load_model().
        sequence: DNA sequence to run, window_bp long.
        window_bp: Length in bp of the model input window.
        win_start: Coordinate of sequence position 0.
        crop_start: Start coordinate to keep (inclusive).
        crop_end: End coordinate to keep (exclusive).
        embedding_crop: Crop embeddings to [crop_start, crop_end); False keeps
            the whole window.
        poolings: Reductions to apply to each embedding tensor. Empty stores
            the tensor itself.
        outputs: How to aggregate the prediction tensors. None stores every
            coverage head raw (cropped) instead - the escape hatch for
            --output_type none. Whenever outputs is given, scope is exactly
            output_heads: a coverage head not in it is dropped, not stored raw.
        output_heads: Which prediction heads to aggregate.
        embedding_layers: Names of the embedding tensors to keep (e.g. FINAL_EMBEDDINGS);
            None keeps every captured one. Predictions are unaffected.

    Returns:
        {tensor_name: (array, attrs)}, embeddings of shape (positions, channels)
        unless pooled, aggregates of shape (tracks,).
    """
    import jax
    from alphagenome_research.model import dna_model as local_dna_model

    dna_sequence = jax.device_put(
        np.asarray(model.one_hot_encoder.encode(sequence))[np.newaxis], model.device
    )
    organism_index = jax.device_put(
        np.full((1,), local_dna_model.convert_to_organism_index(model.organism), dtype=np.int32),
        model.device,
    )
    captured, _ = model.forward_fn(
        model.params, model.state, None, dna_sequence, organism_index
    )

    results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    for name, tensor in captured.items():
        length = tensor.shape[1]
        # Every AlphaGenome tensor spans the full input window, so resolution
        # follows from its length and the origin is the window start.
        resolution = window_bp // length

        head = _prediction_head(name)
        if head is None and embedding_layers is not None and name not in embedding_layers:
            continue
        if head is not None and outputs is not None:
            # Aggregation is active, so the scope is exactly output_heads -
            # a coverage head that was not asked for is dropped here rather
            # than stored raw. Storing it raw-cropped would use the
            # haplotype-shifted crop regardless of embedding_crop (see the
            # branch below), making it a different length per haplotype and
            # breaking the stacked layout's one-shape-per-tensor assumption.
            if head in output_heads:
                results.update(
                    _aggregate(tensor[0], name, head, resolution, win_start, outputs)
                )
            continue

        if embedding_crop or head is not None:
            bin_start, bin_end, attrs = crop_bins(
                length, resolution, win_start, crop_start, crop_end
            )
            if bin_end == bin_start:
                continue
        else:
            bin_start, bin_end = 0, length
            _, _, attrs = crop_bins(
                length, resolution, win_start, win_start, win_start + window_bp
            )

        # Slice on device before converting: a full-window tensor runs to
        # gigabytes and only the reduction is ever kept.
        view = tensor[0, bin_start:bin_end]
        if poolings and head is None:
            for pooling in poolings:
                results[name + pooling.suffix] = (
                    _pool(view, pooling),
                    {**attrs, 'pooled_axis': pooling.axis, 'pooled_func': pooling.func,
                     'pooled_n': view.shape[pooling.dim]},
                )
        else:
            results[name] = (np.asarray(view), attrs)

    return results


def _prediction_head(name: str) -> Optional[str]:
    """'predictions/rna_seq/res_1' -> 'rna_seq'; None for non-prediction tensors."""
    parts = name.split('/')
    return parts[1] if parts[0] == 'predictions' and len(parts) > 1 else None


def _pool(view, pooling: Pooling) -> np.ndarray:
    """Reduce one embedding tensor on device, accumulating in float32."""
    import jax.numpy as jnp

    if pooling.func == 'mean':
        # dtype= accumulates in float32 without materialising a float32 copy
        # of the tensor: uncropped 1bp embeddings are gigabytes in bf16 alone.
        return np.asarray(jnp.mean(view, axis=pooling.dim, dtype=jnp.float32))
    return np.asarray(jnp.max(view, axis=pooling.dim).astype(jnp.float32))


def _aggregate(
    tensor,
    name: str,
    head: str,
    resolution: int,
    win_start: int,
    outputs: OutputAggregation,
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Weighted sum/mean of one prediction tensor over a gene's regions.

    AlphaGenome heads are already per-assay, so no channel subsetting is
    needed; the resolution suffix is kept in the name because a head is
    predicted at both 1bp and 128bp.

    No un-squashing counterpart exists here: unlike Borzoi, AlphaGenome's
    coverage heads are not trained on power-transformed targets, so the raw
    aggregate is already in coverage space.
    """
    import jax.numpy as jnp

    resolution_suffix = name.split('/')[-1]
    base = f'outputs/{head}/{resolution_suffix}'
    n_tracks = tensor.shape[1]
    results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    for request in build_requests(outputs, tensor.shape[0], resolution, win_start):
        attrs = request.attrs(resolution, win_start)

        if request.bin_end == request.bin_start:
            # A single-exon gene has no introns: keep the row so downstream
            # tables stay rectangular, with sum 0 and an undefined mean.
            for func in outputs.funcs:
                empty = np.zeros(n_tracks, dtype=np.float32) if func == 'sum' \
                    else np.full(n_tracks, np.nan, dtype=np.float32)
                results[f'{base}/{request.crop}_{func}'] = (empty, attrs)
            continue

        view = tensor[request.bin_start:request.bin_end].astype(jnp.float32)
        weights = jnp.asarray(request.weights, dtype=jnp.float32)
        weighted = jnp.sum(view * weights[:, None], axis=0)
        total = float(jnp.sum(weights))

        for func in outputs.funcs:
            reduced = weighted if func == 'sum' else weighted / total
            results[f'{base}/{request.crop}_{func}'] = (np.asarray(reduced), attrs)

    return results


def get_track_metadata(model: LoadedModel) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns each coverage head's full track table for the HDF5 /meta group.

    Every column the head's metadata carries is kept (name, ontology curie,
    strand, assay, biosample, ...): it is negligible next to the tensors and
    saves having to re-derive the model's own metadata to read a file later.
    """
    from alphagenome_research.model import heads as heads_module

    metadata: Dict[str, Dict[str, np.ndarray]] = {}
    organism_metadata = model.metadata[model.organism]
    for head_name in COVERAGE_HEADS:
        head_config = heads_module.get_head_config(heads_module.HeadName(head_name))
        df = organism_metadata.get(head_config.output_type)
        if df is None:
            continue
        metadata[head_name] = track_meta.as_arrays(track_meta.build_track_table(df))
    return metadata
