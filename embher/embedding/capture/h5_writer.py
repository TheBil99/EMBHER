"""
HDF5 writers for capture_forward.py, in two layouts.

Every captured tensor is stored as the raw bfloat16 bit pattern in a uint16
dataset (h5py has no native bfloat16 dtype), with a 'dtype' attr of
'bfloat16' recording that, unless --dtype requests float32 storage instead.
Aggregated outputs bypass that policy: they are (n_tracks,) vectors, so
float32 costs nothing, while a bfloat16 sum over a gene body would throw away
precision the aggregation just spent a whole tensor earning.

StackedH5Writer is the layout to use whenever every individual's tensors have
the same shape - pooled embeddings over the uncropped window, and output
aggregates - which is the common case. Sample and haplotype become array axes:

    genes/{ensg}/pooled_embeddings/{tensor}__{axis}_{func}   (n_samples, 2, L)
    genes/{ensg}/outputs/{head}/{res}/{crop}_{func}          (n_samples, 2, n_tracks)

That is ~10^4x fewer HDF5 objects than a group per haplotype, and reading one
tensor across a cohort becomes a single slab read rather than n_samples
dataset opens.

ForwardH5Writer keeps the older group-per-haplotype layout, for the captures
whose arrays are *not* rectangular: --embedding_crop gene and --output_type
none both crop to a haplotype-shifted interval, so their lengths differ by a
bin or two between individuals.

Both write incrementally per (gene, sample, haplotype), so memory stays flat
regardless of how many genes and samples a run covers.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import h5py
import ml_dtypes
import numpy as np

# Attrs that describe where a reduction landed, rather than what it is. They
# differ between individuals whenever an indel resizes a region, so in the
# stacked layout they become (n_samples, 2) columns under genes/{ensg}/coords
# instead of dataset attrs.
COORD_ATTRS = ('bin_start', 'bin_end', 'n_bins', 'n_bp', 'genomic_start', 'genomic_end')


class _BaseH5Writer:
    def __init__(
        self,
        output_path: str,
        dtype: str = 'bfloat16',
        compression: Optional[str] = 'gzip',
        compression_opts: Optional[int] = 4,
        mode: str = 'w',
    ):
        if dtype not in ('bfloat16', 'float32'):
            raise ValueError(f"dtype must be 'bfloat16' or 'float32', got {dtype!r}")
        self.dtype = dtype
        self.compression = compression
        self.compression_opts = compression_opts if compression == 'gzip' else None
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # 'a' (--resume): open what is there rather than truncate it, so
        # completed genes' datasets survive to be skipped and not touched.
        self._file = h5py.File(output_path, mode)

    def _encode(self, array: np.ndarray) -> np.ndarray:
        if self.dtype == 'bfloat16':
            return array.astype(ml_dtypes.bfloat16).view(np.uint16)
        return array.astype(np.float32)

    def write_meta(self, attrs: Dict[str, Any]) -> None:
        meta = self._file.require_group('meta')
        for key, value in attrs.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            meta.attrs[key] = value

    def write_track_metadata(self, track_meta: Dict[str, Dict[str, np.ndarray]]) -> None:
        """
        Store every metadata field each head carries, in channel order.

        Values arrive already typed (see capture.tracks.as_arrays): integer
        track_index, strings everywhere else.
        """
        group = self._file.require_group('meta/tracks')
        for head_name, fields in track_meta.items():
            head_group = group.require_group(head_name)
            for field_name, values in fields.items():
                # --resume reopens a file that already has this: identical
                # content every time (it comes from the checkpoint, not the
                # run), so skip re-creating rather than erroring on it.
                if field_name in head_group:
                    continue
                array = np.asarray(values)
                if array.dtype.kind in ('U', 'O'):
                    array = np.asarray(values, dtype=h5py.string_dtype())
                head_group.create_dataset(field_name, data=array)

    def write_gene(self, ensg: str, attrs: Dict[str, Any]) -> None:
        group = self._file.require_group(f'genes/{ensg}')
        for key, value in attrs.items():
            group.attrs[key] = value

    def mark_gene_complete(self, ensg: str) -> None:
        """Flag a gene as fully written, the unit --resume skips re-doing."""
        self._file.require_group(f'genes/{ensg}').attrs['complete'] = True

    def existing_meta(self) -> Dict[str, Any]:
        """/meta attrs already in the file, or {} for a fresh one. For --resume."""
        return dict(self._file['meta'].attrs) if 'meta' in self._file else {}

    def completed_genes(self) -> set:
        """ensgs already marked complete. For --resume, to skip re-doing them."""
        if 'genes' not in self._file:
            return set()
        return {
            ensg for ensg in self._file['genes']
            if self._file[f'genes/{ensg}'].attrs.get('complete', False)
        }

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class ForwardH5Writer(_BaseH5Writer):
    """One group per (gene, sample, haplotype). Handles ragged arrays."""

    def write_tensor(
        self,
        ensg: str,
        sample_id: str,
        haplotype: str,
        name: str,
        array: np.ndarray,
        attrs: Dict[str, int],
    ) -> None:
        dset = self._file.create_dataset(
            f'genes/{ensg}/{sample_id}/{haplotype}/{name}',
            data=self._encode(array),
            compression=self.compression,
            compression_opts=self.compression_opts,
            chunks=True,
        )
        dset.attrs['dtype'] = self.dtype
        dset.attrs.update(attrs)

    def write_aggregate(
        self,
        ensg: str,
        sample_id: str,
        haplotype: str,
        name: str,
        array: np.ndarray,
        attrs: Dict[str, Any],
    ) -> None:
        """Write one (n_tracks,) aggregate, always float32 regardless of --dtype."""
        dset = self._file.create_dataset(
            f'genes/{ensg}/{sample_id}/{haplotype}/{name}',
            data=np.asarray(array, dtype=np.float32),
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        dset.attrs['dtype'] = 'float32'
        dset.attrs.update(attrs)


class StackedH5Writer(_BaseH5Writer):
    """
    Sample and haplotype as leading array axes, one dataset per gene and tensor.

    Datasets are created on the first haplotype of each gene, from that
    haplotype's shape, and every later write must match it: a shape or frame
    mismatch means the capture is not rectangular after all, and raises rather
    than silently stacking tensors that do not line up.
    """

    def __init__(self, output_path: str, samples: Sequence[str], **kwargs):
        super().__init__(output_path, **kwargs)
        self.samples = list(samples)
        self.haplotypes = ['maternal', 'paternal']

        if 'samples' in self._file:
            # --resume: sample_idx is a position in this list, baked into
            # every dataset already written, so a different order or set
            # would silently misalign old and new data in the same slots.
            existing = [s.decode() for s in self._file['samples'][...]]
            if existing != self.samples:
                raise ValueError(
                    f"'{output_path}' already has {len(existing)} sample(s) that do not "
                    f"match this run's {len(self.samples)}; resuming would misalign the "
                    f"sample axis of every stacked dataset"
                )
        else:
            string = h5py.string_dtype()
            self._file.create_dataset('samples', data=np.array(self.samples, dtype=string))
            self._file.create_dataset(
                'haplotypes', data=np.array(self.haplotypes, dtype=string)
            )

    def write_haplotype(
        self,
        ensg: str,
        sample_idx: int,
        hap_idx: int,
        tensors: Dict[str, Any],
    ) -> None:
        """Write every tensor of one (gene, sample, haplotype) into its slot."""
        for name, (array, attrs) in tensors.items():
            is_aggregate = name.startswith('outputs/')
            path = (
                f'genes/{ensg}/{name}' if is_aggregate
                else f'genes/{ensg}/pooled_embeddings/{name}'
            )
            array = np.asarray(array)

            if path in self._file:
                dset = self._file[path]
                if dset.shape[2:] != array.shape:
                    raise ValueError(
                        f"'{path}' was created with shape {dset.shape[2:]} but "
                        f"sample {self.samples[sample_idx]}/{self.haplotypes[hap_idx]} "
                        f"gives {array.shape}; this capture is not rectangular, so it "
                        f"needs the group-per-haplotype layout (ForwardH5Writer)"
                    )
            else:
                dset = self._create(path, array, attrs, is_aggregate)

            dset[sample_idx, hap_idx] = (
                np.asarray(array, dtype=np.float32) if is_aggregate else self._encode(array)
            )
            self._record_frame(ensg, sample_idx, hap_idx, attrs, dset)

    def _create(self, path: str, array: np.ndarray, attrs: Dict[str, Any], is_aggregate: bool):
        shape = (len(self.samples), 2) + array.shape
        dset = self._file.create_dataset(
            path,
            shape=shape,
            dtype=np.float32 if is_aggregate else self._encode(array[:0]).dtype,
            # One chunk per haplotype: each write is a single whole-chunk write,
            # with no read-modify-write against the compressor, and a read
            # across the cohort walks the chunks in order.
            chunks=(1, 2) + array.shape,
            compression=self.compression,
            compression_opts=self.compression_opts,
            # bfloat16 bit patterns compress poorly byte-interleaved; splitting
            # high and low bytes gives the deflater runs to work with.
            shuffle=self.compression is not None,
            fillvalue=np.nan if is_aggregate else 0,
        )
        dset.attrs['dtype'] = 'float32' if is_aggregate else self.dtype
        dset.attrs.update({k: v for k, v in attrs.items() if k not in COORD_ATTRS})
        return dset

    def _record_frame(self, ensg, sample_idx, hap_idx, attrs, dset) -> None:
        """
        Per-haplotype placement goes to frames/; constant placement stays an attr.

        An aggregate's region moves and resizes with each haplotype's indels -
        'n_bp' is the denominator of every mean - so those go to a column keyed
        by the crop. A pooled embedding over the uncropped window has the same
        frame for everyone, so it keeps its attrs, and a disagreement means the
        rectangular assumption is broken.
        """
        coords = {k: v for k, v in attrs.items() if k in COORD_ATTRS}
        crop = attrs.get('crop')
        if crop is None:
            for key, value in coords.items():
                if key in dset.attrs and dset.attrs[key] != value:
                    raise ValueError(
                        f"'{dset.name}' has {key}={value} for "
                        f"{self.samples[sample_idx]}/{self.haplotypes[hap_idx]} but "
                        f"{dset.attrs[key]} for the first haplotype; this capture is "
                        f"not rectangular and needs the group-per-haplotype layout"
                    )
                dset.attrs[key] = value
            return
        self.write_coords(ensg, sample_idx, hap_idx, coords, subgroup=crop)

    def write_coords(
        self,
        ensg: str,
        sample_idx: int,
        hap_idx: int,
        values: Dict[str, float],
        subgroup: str = '',
    ) -> None:
        """Record one (n_samples, 2) per-haplotype scalar under genes/{ensg}/coords."""
        base = f'genes/{ensg}/coords' + (f'/{subgroup}' if subgroup else '')
        for key, value in values.items():
            path = f'{base}/{key}'
            if path in self._file:
                dset = self._file[path]
            else:
                # float64 throughout: it holds fractional base-pair counts and
                # is exact for genomic coordinates well past the genome's size.
                dset = self._file.create_dataset(
                    path, shape=(len(self.samples), 2), dtype='f8', fillvalue=np.nan
                )
            dset[sample_idx, hap_idx] = value


def merge_stacked_delta(seed_path: str, delta_path: str, output_path: str, ensg: str) -> str:
    """
    Combine a --seed_h5 (already-captured samples) with a delta file holding
    only the newly-captured samples for one gene, and write their union to
    output_path.

    Every seed row is copied verbatim and keeps its position; delta rows are
    appended after, in delta's own sample order. Nothing already in the seed
    is ever re-read from the model or reordered - this is the only place a
    --seed_h5 run touches previously-captured data at all. Writes to a
    '.merging' temp path first and os.replace()s it over output_path, so a
    crash mid-merge never leaves a half-written file at the real path.

    Raises rather than guessing whenever seed and delta disagree in a way
    that would make the union ambiguous (overlapping samples, a dataset
    missing from one side, mismatched per-dataset shape) - the caller
    (capture_forward.py's --seed_h5 path) is expected to have already
    checked the coarser meta/region compatibility before calling this.
    """
    import os

    tmp_path = f'{output_path}.merging'
    with h5py.File(seed_path, 'r') as sf, h5py.File(delta_path, 'r') as df, \
            h5py.File(tmp_path, 'w') as of:
        if 'samples' not in sf or f'genes/{ensg}' not in sf:
            raise ValueError(f"seed '{seed_path}' has no samples/genes/{ensg} to merge into")
        if 'samples' not in df or f'genes/{ensg}' not in df:
            raise ValueError(f"delta '{delta_path}' has no samples/genes/{ensg} to merge")

        seed_samples = [s.decode() for s in sf['samples'][...]]
        delta_samples = [s.decode() for s in df['samples'][...]]
        overlap = set(seed_samples) & set(delta_samples)
        if overlap:
            raise ValueError(
                f"seed and delta both already have sample(s) {sorted(overlap)}; "
                f"refusing to merge duplicates into '{output_path}'"
            )
        n_seed, n_delta = len(seed_samples), len(delta_samples)
        merged_samples = seed_samples + delta_samples

        of.create_dataset('samples', data=np.array(merged_samples, dtype=h5py.string_dtype()))
        sf.copy(sf['haplotypes'], of, name='haplotypes')
        sf.copy(sf['meta'], of, name='meta')

        seed_gene = sf[f'genes/{ensg}']
        out_gene = of.require_group(f'genes/{ensg}')
        for key, value in seed_gene.attrs.items():
            out_gene.attrs[key] = value

        delta_gene = df[f'genes/{ensg}']

        def merge_into(seed_group, delta_group, out_group):
            for name, item in seed_group.items():
                if isinstance(item, h5py.Group):
                    if name not in delta_group or not isinstance(delta_group[name], h5py.Group):
                        raise ValueError(
                            f"'genes/{ensg}/{item.name.split(ensg, 1)[-1]}' is a group in the "
                            f"seed but missing (or not a group) in the delta; schemas disagree"
                        )
                    merge_into(item, delta_group[name], out_group.require_group(name))
                    continue
                if name not in delta_group or not isinstance(delta_group[name], h5py.Dataset):
                    raise ValueError(
                        f"'{item.name}' exists in the seed but not the delta; the two files "
                        f"were not captured with the same settings"
                    )
                delta_item = delta_group[name]
                if item.shape[2:] != delta_item.shape[2:]:
                    raise ValueError(
                        f"'{item.name}' has shape {item.shape[2:]} in the seed but "
                        f"{delta_item.shape[2:]} in the delta; refusing to merge mismatched tensors"
                    )
                merged = np.concatenate([item[...], delta_item[...]], axis=0)
                tail = item.shape[2:]
                dset = out_group.create_dataset(
                    name,
                    data=merged,
                    chunks=(1,) + item.shape[1:] if tail else True,
                    compression=item.compression,
                    compression_opts=item.compression_opts,
                    shuffle=item.shuffle,
                    fillvalue=item.fillvalue,
                )
                for key, value in item.attrs.items():
                    dset.attrs[key] = value

        merge_into(seed_gene, delta_gene, out_gene)

    os.replace(tmp_path, output_path)
    return output_path
