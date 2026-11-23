"""
Early-exit Borzoi forward pass, layer by layer.

This mirrors the Borzoi forward pass used for capture (capture/borzoi.py), so a layer here is
exactly the tensor that capture pools for `<layer>:pos` / `<layer>:chan`. It differs in two
ways, both needed for attribution rather than extraction:

  * it stops at the requested layer, so the backward pass never stores the activations
    of layers that cannot affect the target;
  * each stage can run under torch.utils.checkpoint, trading recompute for the memory a
    524 kb backward pass otherwise needs.

Layer names follow the extraction script: stem, res_tower_{1..n}, unet1,
transformer_{1..n}, upsample1, separable1, upsample0, separable0, trunk, joined,
human_head. Tensors come back channels-first, (B, C, L'), L' being the bin count at that
stage (524288 / 2^k up to unet1, then the crop leaves 6144 bins from `trunk` on).
"""

from typing import Callable, List

import torch
from torch.utils.checkpoint import checkpoint

from embher.embedding.attribution.capture_layers import CAPTURE_LAYERS, capture_label  # noqa: F401


# Layers of Borzoi (5 res_tower convs, 8 transformer blocks), in forward order. Documentation and
# `--list_layers` only: the authoritative list is layer_names(trunk), read off the model.
BORZOI_LAYERS = (
    ["stem"] + [f"res_tower_{i}" for i in range(1, 6)] + ["unet1"]
    + [f"transformer_{i}" for i in range(1, 9)]
    + ["upsample1", "separable1", "upsample0", "separable0", "trunk", "joined", "human_head"]
)


def layer_names(trunk) -> List[str]:
    """Architecture-depth checkpoint list in forward order, read off the model itself."""
    n_res = sum(1 for i, _ in enumerate(trunk.res_tower) if i % 2 == 0)
    n_xfmr = len(trunk.transformer)
    names = ["stem"]
    names += [f"res_tower_{i + 1}" for i in range(n_res)]
    names += ["unet1"]
    names += [f"transformer_{i + 1}" for i in range(n_xfmr)]
    names += ["upsample1", "separable1", "upsample0", "separable0", "trunk", "joined",
              "human_head"]
    return names


def pool(t: torch.Tensor, axis: str) -> torch.Tensor:
    """Pool a (B, C, L') layer tensor exactly as the extraction script does, in float32.

    `pos` averages over positions (-> (B, C), the pooled embedding);
    `chan` averages over channels (-> (B, L')).
    """
    if axis == "pos":
        return t.mean(dim=-1, dtype=torch.float32)
    if axis == "chan":
        return t.mean(dim=1, dtype=torch.float32)
    raise ValueError(f"axis must be 'pos' or 'chan', got {axis!r}")


def run_to_layer(trunk, x: torch.Tensor, layer: str, use_checkpoint: bool = False
                 ) -> torch.Tensor:
    """Forward `x` (B, 4, L) through `trunk` and return the activation at `layer`."""
    names = layer_names(trunk)
    if layer not in names:
        raise ValueError(f"unknown layer {layer!r}; one of {names}")

    def run(fn: Callable, *args):
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    x = run(trunk.conv_dna, x)
    if layer == "stem":
        return x

    conv_idx = 0
    for i, block in enumerate(trunk.res_tower):
        x = run(block, x)
        if i % 2 == 0:
            conv_idx += 1
            if layer == f"res_tower_{conv_idx}":
                return x
    x_unet0 = x

    x_unet1 = run(trunk.unet1, x_unet0)
    if layer == "unet1":
        return x_unet1

    xt = trunk._max_pool(x_unet1).permute(0, 2, 1)   # (B, L, C) for the transformer
    for i, block in enumerate(trunk.transformer):
        xt = run(block, xt)
        if layer == f"transformer_{i + 1}":
            return xt.permute(0, 2, 1)
    xt = xt.permute(0, 2, 1)

    x_unet1p = trunk.horizontal_conv1(x_unet1)
    x_unet0p = trunk.horizontal_conv0(x_unet0)
    x = trunk.upsampling_unet1(xt) + x_unet1p
    if layer == "upsample1":
        return x
    x = run(trunk.separable1, x)
    if layer == "separable1":
        return x
    x = trunk.upsampling_unet0(x) + x_unet0p
    if layer == "upsample0":
        return x
    x = run(trunk.separable0, x)
    if layer == "separable0":
        return x

    x = trunk.crop(x.permute(0, 2, 1)).permute(0, 2, 1)
    if layer == "trunk":
        return x
    joined = trunk.final_joined_convs(x)
    if layer == "joined":
        return joined

    # human_head: same fp32 head as the extraction script, outside any autocast region.
    with torch.amp.autocast("cuda", enabled=False):
        return trunk.final_softplus(trunk.human_head(joined.float()))


# How a layer is computed, stored with its attribution so results of an older definition are
# recomputed, not reused: "v1" until the upsample layers were fixed to include the skip connection.
_LAYER_DEFINITION = {"decoder/upsample1": "post_skip", "decoder/upsample0": "post_skip"}


def layer_definition(layer: str) -> str:
    return _LAYER_DEFINITION.get(layer, "v1")


def run_to_capture(trunk, x: torch.Tensor, layer: str, use_checkpoint: bool = False
                   ) -> torch.Tensor:
    """Forward `x` (B, 4, L) and return the activation capture_forward.py stores as `layer`,
    channels-first (B, C, L') whatever the layer's own layout."""
    if layer not in CAPTURE_LAYERS:
        raise ValueError(f"unknown capture layer {layer!r}; one of {list(CAPTURE_LAYERS)}")

    def run(fn: Callable, *args):
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    x = run(trunk.conv_dna, x)
    if layer == "encoder/conv_dna":
        return x
    for i, block in enumerate(trunk.res_tower):
        x = run(block, x)
        if i % 2 == 0 and layer == f"encoder/res_tower_{i // 2}":
            return x
    x_unet0 = x
    x_unet1 = run(trunk.unet1, x_unet0)
    if layer == "encoder/unet1":
        return x_unet1
    xt = trunk._max_pool(x_unet1).permute(0, 2, 1)         # (B, L, C), the transformer input
    if layer == "encoder/trunk":
        return xt.permute(0, 2, 1)
    h1 = trunk.horizontal_conv1(x_unet1)
    if layer == "skip/horizontal_conv1":
        return h1
    h0 = trunk.horizontal_conv0(x_unet0)
    if layer == "skip/horizontal_conv0":
        return h0
    for i, block in enumerate(trunk.transformer):
        xt = run(block, xt)
        if layer == f"transformer/block_{i}":
            return xt.permute(0, 2, 1)
    xt = xt.permute(0, 2, 1)
    up1 = trunk.upsampling_unet1(xt)
    x = up1 + h1                                            # capture's upsample1 includes the skip
    if layer == "decoder/upsample1":
        return x
    x = run(trunk.separable1, x)
    if layer == "decoder/separable1":
        return x
    up0 = trunk.upsampling_unet0(x)
    x = up0 + h0                                            # capture's upsample0 includes the skip
    if layer == "decoder/upsample0":
        return x
    x = run(trunk.separable0, x)
    if layer == "decoder/separable0":
        return x
    x = trunk.crop(x.permute(0, 2, 1)).permute(0, 2, 1)
    if layer == "embeddings/cropped":
        return x
    return trunk.final_joined_convs(x)                      # embeddings/final
