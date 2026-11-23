"""
Uniform track metadata for both backends.

Every column the source metadata carries is written out: it is a few hundred
kilobytes at most, and having it in the file means never having to hunt down
the matching targets/metadata table to interpret an aggregate months later.

On top of those columns, two fields are always present:

  track_index  channel index this row describes. For a fused head that is
               subset (Borzoi keeps only its RNA tracks out of 7611) the
               aggregate's column j is *not* channel j, so this is the only
               key that survives subsetting.
  track_name   a unique human-readable name, uniqueness enforced rather than
               assumed - Borzoi's `identifier` already carries the strand as a
               '+'/'-' suffix, AlphaGenome's naming is checked at run time,
               and any collision that survives is disambiguated visibly
               instead of silently mislabelling a column.

AlphaGenome and Borzoi share no track identifier, so joining the two models'
tracks stays a deliberate downstream step; this only guarantees each file
describes its own columns completely.
"""

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Candidate columns for the readable name, best first.
_NAME_COLUMNS = ('identifier', 'name', 'description', 'track_id')
_STRAND_COLUMNS = ('strand', 'Strand')


def _sanitise(field: str) -> str:
    """HDF5 dataset names cannot contain '/'."""
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(field)).strip('_') or 'field'


def _unique_names(names: Sequence[str], indices: Sequence[int]) -> List[str]:
    """Append '#<track_index>' to any name that appears more than once."""
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return [
        f'{name}#{index}' if counts[name] > 1 else name
        for name, index in zip(names, indices)
    ]


def build_track_table(
    df: pd.DataFrame,
    track_index: Optional[Sequence[int]] = None,
    name_column: Optional[str] = None,
) -> Dict[str, list]:
    """
    Turn a track metadata DataFrame into the field->values dict the writer stores.

    Args:
        df: One row per track, in channel order.
        track_index: Channel indices to keep, in output order. None keeps all.
        name_column: Column to build track_name from; None picks the first of
            identifier/name/description/track_id that exists.

    Returns:
        {field: values}, every list the same length as the kept tracks.
        'track_index' is integer; every other field is string.
    """
    if track_index is None:
        selected = df.reset_index(drop=True)
        indices = list(range(len(selected)))
    else:
        indices = [int(i) for i in track_index]
        selected = df.iloc[indices].reset_index(drop=True)

    if name_column is None:
        name_column = next((c for c in _NAME_COLUMNS if c in selected.columns), None)

    if name_column is None:
        names = [f'track_{i}' for i in indices]
    else:
        names = selected[name_column].astype(str).tolist()
        strand_column = next((c for c in _STRAND_COLUMNS if c in selected.columns), None)
        if strand_column is not None:
            strands = selected[strand_column].astype(str).tolist()
            # Borzoi's identifier already ends in '+'/'-'; don't say it twice.
            names = [
                name if (not strand or strand in ('nan', 'None') or name.endswith(strand))
                else f'{name}{strand}'
                for name, strand in zip(names, strands)
            ]

    table: Dict[str, list] = {
        'track_index': indices,
        'track_name': _unique_names(names, indices),
    }
    for column in selected.columns:
        field = _sanitise(column)
        if field not in table:
            table[field] = selected[column].astype(str).tolist()

    assert len(set(table['track_name'])) == len(table['track_name']), \
        'track_name must be unique after disambiguation'
    return table


def assert_channel_count(table: Dict[str, list], n_channels: int, head: str) -> None:
    """
    Check metadata rows against the tensor's channel count.

    Both backends take the metadata's row order as the channel order. That is
    almost certainly right, but a mismatch would mislabel every column of every
    aggregate, so it fails loudly here rather than silently in the file.
    """
    n_rows = len(table['track_index'])
    if n_rows != n_channels:
        raise ValueError(
            f"track metadata for '{head}' has {n_rows} rows but the tensor has "
            f"{n_channels} channels; channel order cannot be trusted"
        )


def as_arrays(table: Dict[str, list]) -> Dict[str, np.ndarray]:
    """Numpy views for the writer: integer track_index, strings elsewhere."""
    import h5py

    arrays = {}
    for field, values in table.items():
        if field == 'track_index':
            arrays[field] = np.asarray(values, dtype=np.int32)
        else:
            arrays[field] = np.asarray(values, dtype=h5py.string_dtype())
    return arrays
