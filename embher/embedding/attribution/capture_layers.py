"""The 24 Borzoi checkpoints capture_forward.py stores, in architecture-depth order. Plain Python (no torch), so figure
scripts can read it without the model stack: they add this folder to sys.path and import it directly."""

# The 24 checkpoints stored per gene (capture/borzoi.py's _capture_specs), in architecture-depth
# order.
# `borzoi_forward.run_to_capture` returns what those hooks stored, which differs from `run_to_layer` in two
# places: the decoder `upsample` layers include the skip connection (borzoi-pytorch adds it in place,
# `x += skip`, so the tensor the hook kept is the sum), and `encoder/trunk` (the transformer input),
# the two `skip/horizontal_conv` outputs and `embeddings/cropped` exist here too.
CAPTURE_LAYERS = (
    ["encoder/conv_dna"] + [f"encoder/res_tower_{i}" for i in range(5)]
    + ["encoder/unet1", "encoder/trunk"] + [f"transformer/block_{i}" for i in range(8)]
    + ["skip/horizontal_conv1", "skip/horizontal_conv0", "decoder/upsample1", "decoder/separable1",
       "decoder/upsample0", "decoder/separable0", "embeddings/cropped", "embeddings/final"]
)


def capture_label(layer: str) -> str:
    """'encoder/res_tower_0' -> 'encoder_res_tower_0', the label workflows 13/99 use."""
    return layer.replace("/", "_")
