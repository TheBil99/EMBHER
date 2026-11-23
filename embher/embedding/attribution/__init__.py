"""
Sequence attribution for Borzoi (input x gradient and friends) on a reference window.

Two kinds of scalar are differentiated with respect to the one-hot input:

  * ExpressionTarget - the expression prediction: chosen head tracks,
    each unsquashed with its own scale, summed over the gene's exon bins, log2(x + 1) per
    track, averaged over tracks (the extraction pipeline's scalar_prediction).
  * EmbeddingTarget  - a layer's pooled embedding (default joined:pos) reduced to one number by a readout (see targets.readouts).

Results are wrapped in decima's `Attribution` (peaks, seqlets, motifs, bigwig, plots), so
gReLU/Decima do the post-processing; this package only supplies the differentiable
scalar, the Captum driver and the I/O.

Nothing is imported here: torch, captum, grelu and decima are heavy and the submodules
pull in only what they need (cf. utils/capture/__init__.py).
"""
