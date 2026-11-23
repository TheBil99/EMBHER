"""
Instrumented Borzoi forward passes that capture intermediate activations.

Borzoi is a plain torch.nn.Module, so capture uses register_forward_hook -
the PyTorch counterpart of the hk.intercept_methods approach in
capture/alphagenome.py. Captured per forward pass (see Borzoi.get_embs_after_crop):

  encoder/conv_dna, encoder/res_tower_{0..4}, encoder/unet1, encoder/trunk
  transformer/block_{0..7}
  skip/horizontal_conv{0,1}
  decoder/upsample{0,1}, decoder/separable{0,1}
  embeddings/cropped, embeddings/final
  predictions/human

Two Borzoi-specific wrinkles drive the design:

  * `self.crop` (TargetLengthCrop) center-crops 16384 bins down to 6144 before
    the head, so everything from `crop` onward covers only the middle of the
    input window - its resolution can't be derived from length the way the
    earlier tensors' can.
  * `self._max_pool` and `self.upsample` are single module instances reused at
    several points in the graph, so hooking them would fire repeatedly with
    different shapes. Only uniquely-positioned modules are hooked; the trunk
    is read off the first transformer block's input instead.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from embher.embedding.utils.borzoi_utils import BORZOI_SEQ_LEN, dna_to_onehot
from embher.embedding.capture import tracks as track_meta
from embher.embedding.capture.crop import crop_bins
from embher.embedding.capture.reduce import OutputAggregation, Pooling, build_requests

# Borzoi is trained on coverage squashed by a power transform above a soft
# clip, so its outputs live in squashed space. These are baskerville's
# defaults, overridden per track wherever the targets table carries them.
DEFAULT_TRACK_SCALE = 0.01
DEFAULT_CLIP_SOFT = 384.0
DEFAULT_TRACK_TRANSFORM = 0.75

# The embedding feeding the output head (final_joined_convs' 1920-dim output), the one
# capture_forward.py's --embedding_layers final keeps.
FINAL_EMBEDDINGS = ('embeddings/final',)

# res_tower interleaves ConvBlocks with the shared max-pool; these are the
# ConvBlocks.
_RES_TOWER_CONV_INDICES = (0, 2, 4, 6, 8)


@dataclass(frozen=True)
class _Capture:
    name: str
    # The module to hook, or None when the tensor is read from elsewhere.
    module: Any = None
    # Tensors after `self.crop` cover only the centre of the input window.
    post_crop: bool = False
    # The transformer runs channels-last; the conv stages run channels-first.
    channels_last: bool = False


# Captured without a hook of their own: the trunk off the first transformer
# block's input, the predictions off the forward's return value.
_UNHOOKED = (
    _Capture('encoder/trunk', channels_last=True),
    _Capture('predictions/human', post_crop=True),
)


def _capture_specs(model) -> List[_Capture]:
    specs = [_Capture('encoder/conv_dna', model.conv_dna)]
    specs += [
        _Capture(f'encoder/res_tower_{i}', model.res_tower[idx])
        for i, idx in enumerate(_RES_TOWER_CONV_INDICES)
    ]
    specs.append(_Capture('encoder/unet1', model.unet1))
    specs += [
        _Capture(f'transformer/block_{i}', block, channels_last=True)
        for i, block in enumerate(model.transformer)
    ]
    specs += [
        _Capture('skip/horizontal_conv1', model.horizontal_conv1),
        _Capture('skip/horizontal_conv0', model.horizontal_conv0),
        _Capture('decoder/upsample1', model.upsampling_unet1),
        _Capture('decoder/separable1', model.separable1),
        _Capture('decoder/upsample0', model.upsampling_unet0),
        _Capture('decoder/separable0', model.separable0),
        # get_embs_after_crop calls self.crop(x.permute(0,2,1)) and permutes
        # back afterwards, so this hook - unlike every other conv-stage hook -
        # observes a channels-last tensor.
        _Capture('embeddings/cropped', model.crop, post_crop=True, channels_last=True),
        _Capture('embeddings/final', model.final_joined_convs, post_crop=True),
    ]
    return specs


def run_forward(
    model,
    sequence: str,
    window_bp: int,
    win_start: int,
    crop_start: int,
    crop_end: int,
    embedding_crop: bool = True,
    poolings: Sequence[Pooling] = (),
    outputs: Optional[OutputAggregation] = None,
    unsquash_params: Optional[Dict[str, Any]] = None,
    embedding_layers: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Run one instrumented Borzoi forward pass, reducing tensors before transfer.

    Pooling and aggregation run on device deliberately: an uncropped tensor is
    hundreds of megabytes and only its reduction is ever kept, so reducing
    after the .cpu() would move the very data the flags exist to avoid.

    Args:
        model: Borzoi/AnnotatedBorzoi from init_borzoi_model().
        sequence: DNA sequence to run.
        window_bp: Length in bp of the model input window (BORZOI_SEQ_LEN).
        win_start: Coordinate of sequence position 0.
        crop_start: Start coordinate to keep (inclusive).
        crop_end: End coordinate to keep (exclusive).
        embedding_crop: Crop embeddings to [crop_start, crop_end); False keeps
            the whole window.
        poolings: Reductions to apply to each embedding tensor. Empty stores
            the tensor itself.
        outputs: How to aggregate the prediction tensors. None stores them raw.
        unsquash_params: Per-track scale/clip_soft/transform for un-squashing,
            aligned to outputs.track_index.
        embedding_layers: Names of the embedding tensors to keep (e.g. FINAL_EMBEDDINGS);
            None keeps every captured one. Predictions are unaffected.

    Returns:
        {tensor_name: (array, attrs)}. Tensors whose coverage misses the
        requested region entirely are omitted.
    """
    import torch

    device = next(model.parameters()).device
    captured: Dict[str, Any] = {}
    specs = _capture_specs(model)
    handles = []

    def _make_hook(spec: _Capture):
        def hook(module, inputs, output):
            captured[spec.name] = output
            if spec.name == 'transformer/block_0':
                # The pooled post-unet1 trunk is this block's input; the shared
                # max-pool that produces it cannot be hooked safely.
                captured['encoder/trunk'] = inputs[0]
        return hook

    for spec in specs:
        handles.append(spec.module.register_forward_hook(_make_hook(spec)))

    try:
        x = dna_to_onehot(sequence, BORZOI_SEQ_LEN).to(device)
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    predictions = model(x, is_human=True)
            else:
                predictions = model(x, is_human=True)
    finally:
        for handle in handles:
            handle.remove()

    captured['predictions/human'] = predictions

    layout = {spec.name: spec for spec in list(specs) + list(_UNHOOKED)}
    # Resolution of the tensors feeding the crop, needed to place everything
    # downstream of it; read off the crop's own output rather than assuming
    # the default bins_to_return.
    crop_input_length = captured['decoder/separable0'].shape[-1]
    crop_resolution = window_bp // crop_input_length

    results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
    for name, tensor in captured.items():
        spec = layout[name]

        tensor = tensor[0]
        if not spec.channels_last:
            tensor = tensor.transpose(0, 1)
        length = tensor.shape[0]

        if spec.post_crop:
            resolution = crop_resolution
            tensor_start = win_start + ((crop_input_length - length) // 2) * resolution
        else:
            resolution = window_bp // length
            tensor_start = win_start

        is_prediction = name.startswith('predictions/')
        if not is_prediction and embedding_layers is not None and name not in embedding_layers:
            continue
        if is_prediction and outputs is not None:
            results.update(
                _aggregate(tensor, resolution, tensor_start, outputs, unsquash_params)
            )
            continue

        # Predictions that are not aggregated are still cropped and stored as
        # tracks: pooling over the track axis of a coverage head is meaningless.
        if embedding_crop or is_prediction:
            bin_start, bin_end, attrs = crop_bins(
                length, resolution, tensor_start, crop_start, crop_end
            )
            if bin_end == bin_start:
                continue
        else:
            bin_start, bin_end = 0, length
            _, _, attrs = crop_bins(
                length, resolution, tensor_start,
                tensor_start, tensor_start + length * resolution,
            )

        # Slice on device before converting: a full-window tensor runs to
        # hundreds of megabytes and only the reduction is ever kept.
        view = tensor[bin_start:bin_end]
        if poolings and not is_prediction:
            for pooling in poolings:
                results[name + pooling.suffix] = (
                    _pool(view, pooling),
                    {**attrs, 'pooled_axis': pooling.axis, 'pooled_func': pooling.func,
                     'pooled_n': view.shape[pooling.dim]},
                )
        else:
            results[name] = (view.to(torch.float32).cpu().numpy(), attrs)

    return results


def _pool(view, pooling: Pooling) -> np.ndarray:
    """Reduce one embedding tensor on device, accumulating in float32."""
    import torch

    if pooling.func == 'mean':
        # dtype= accumulates in float32 without materialising a float32 copy
        # of the tensor, which for an uncropped window is hundreds of MB.
        reduced = torch.mean(view, dim=pooling.dim, dtype=torch.float32)
    else:
        reduced = view.max(dim=pooling.dim).values.to(torch.float32)
    return reduced.cpu().numpy()


def _undo_squash(values, params: Dict[str, Any]):
    """
    Invert Borzoi's squashed coverage scaling, per track.

    Aggregating in squashed space is not equivalent to aggregating raw
    coverage and squashing afterwards - the transform is nonlinear, so the
    order cannot be swapped downstream once bins are collapsed. Both spaces
    are therefore computed here, from the same forward pass.
    """
    import torch

    def as_tensor(key, default):
        value = params.get(key, default)
        return torch.as_tensor(value, device=values.device, dtype=torch.float32)

    scale = as_tensor('scale', DEFAULT_TRACK_SCALE)
    clip_soft = as_tensor('clip_soft', DEFAULT_CLIP_SOFT)
    transform = as_tensor('transform', DEFAULT_TRACK_TRANSFORM)

    unscaled = values / scale
    unclipped = torch.where(
        unscaled > clip_soft, (unscaled - clip_soft) ** 2 + clip_soft, unscaled
    )
    return unclipped.clamp(min=0.0) ** (1.0 / transform)


def _aggregate(
    tensor,
    resolution: int,
    tensor_start: int,
    outputs: OutputAggregation,
    unsquash_params: Optional[Dict[str, Any]],
) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Weighted sum/mean of one prediction tensor over a gene's regions.

    Borzoi fuses all 7611 human tracks into one head, so the requested subset
    is selected here; the aggregate's column j is then not channel j, which is
    why /meta/tracks records track_index.
    """
    import torch

    if outputs.track_index is not None:
        index = torch.as_tensor(
            np.asarray(outputs.track_index), device=tensor.device, dtype=torch.long
        )
        tensor = tensor.index_select(1, index)

    n_tracks = tensor.shape[1]
    results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}

    # Borzoi's outputs live in squashed space; AlphaGenome's heads are not
    # power-transformed at all, so there is no squashed/unsquashed duality to
    # name for them and their aggregates keep the bare '{crop}_{func}' name.
    spaces = ['_squashed', '_unsquashed'] if (outputs.unsquash and unsquash_params) else ['']

    for request in build_requests(outputs, tensor.shape[0], resolution, tensor_start):
        attrs = request.attrs(resolution, tensor_start)

        if request.bin_end == request.bin_start:
            # A single-exon gene has no introns: keep the row so downstream
            # tables stay rectangular, with sum 0 and an undefined mean.
            for space in spaces:
                for func in outputs.funcs:
                    empty = np.zeros(n_tracks, dtype=np.float32) if func == 'sum' \
                        else np.full(n_tracks, np.nan, dtype=np.float32)
                    results[f'outputs/rna_seq/{request.crop}_{func}{space}'] = (empty, attrs)
            continue

        view = tensor[request.bin_start:request.bin_end].to(torch.float32)
        weights = torch.as_tensor(request.weights, device=view.device, dtype=torch.float32)
        total = float(weights.sum())

        for space in spaces:
            values = _undo_squash(view, unsquash_params) if space == '_unsquashed' else view
            weighted = (values * weights[:, None]).sum(dim=0)
            for func in outputs.funcs:
                reduced = weighted if func == 'sum' else weighted / total
                results[f'outputs/rna_seq/{request.crop}_{func}{space}'] = (
                    reduced.cpu().numpy(), attrs
                )

    return results


def get_track_metadata(
    tracks_df: pd.DataFrame,
    track_index: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns Borzoi's full track table for the HDF5 /meta group.

    Every column the targets table carries is kept - it is tiny next to the
    tensors, and it saves having to re-find the matching targets file to
    interpret an aggregate later. When an RNA subset is aggregated, its own
    table is written alongside the full one, carrying the track_index that
    maps its columns back to the 7611 channels.
    """
    meta = {'human': track_meta.as_arrays(track_meta.build_track_table(tracks_df))}
    if track_index is not None:
        meta['rna_seq'] = track_meta.as_arrays(
            track_meta.build_track_table(tracks_df, track_index=track_index)
        )
    return meta
