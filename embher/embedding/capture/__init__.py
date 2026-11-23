"""
Forward-pass capture: dump a model's intermediate and final tensors to HDF5.

Both backends expose the same three entry points, so capture_forward.py only
branches when loading the model:

    alphagenome.load_model(weights_path) -> model
    <backend>.run_forward(model, sequence, window_bp, win_start, crop_start,
                          crop_end, embedding_crop, poolings, outputs)
    <backend>.get_track_metadata(...)

run_forward returns {tensor_name: (array, attrs)}, where `attrs` describes the
region the array covers. A tensor is omitted when its coverage misses the
region entirely - Borzoi centre-crops before its head, so its final tensors
only span the middle of the input window.

Three kinds of array come back, and both backends reduce on device, before the
transfer, since an uncropped tensor is gigabytes and only its reduction is kept:

  * embeddings, (positions, channels), optionally cropped to the gene body
  * pooled embeddings, mean/max over one axis (see capture.reduce)
  * output aggregates, (tracks,), weighted sum/mean of a prediction head over
    a gene's exonic/intronic/whole-body coordinates in one haplotype's frame

Results are written by capture.h5_writer, which stacks sample and haplotype
into array axes whenever every individual's tensors share a shape, and falls
back to a group per haplotype for the cropped (ragged) captures.

The supporting modules are all import-light: capture.reduce (reduction specs
and bin weights), capture.regions (exon/intron intervals from GENCODE),
capture.tracks (uniform, uniquely-named track metadata), capture.crop
(region-to-bin mapping).

Nothing is imported here: the two backends pull in disjoint heavy stacks
(jax/haiku/alphagenome_research vs torch/borzoi_pytorch), so importing this
package must not drag in either.
"""
