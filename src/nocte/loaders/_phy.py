"""Loading of Phy-compatible spike-sorting output."""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

import nocte._coll.trains
import nocte._coll.windows
import nocte._core.sampling

logger = logging.getLogger(__name__)


def _load_int_vector(
    path: pathlib.Path,
) -> np.ndarray:
    """Load a one-dimensional integer NumPy array."""
    values = np.asarray(
        np.load(
            path,
            allow_pickle=False,
        )
    )

    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    elif values.ndim != 1:
        raise ValueError(f'{path.name} must be one-dimensional')

    if not np.issubdtype(
        values.dtype,
        np.integer,
    ):
        raise TypeError(f'{path.name} must contain integers')

    return values.astype(
        np.int64,
        copy=False,
    )


def _load_cluster_meta(
    folder: pathlib.Path,
    unit_ids: pd.Index,
) -> pd.DataFrame:
    """
    Load authoritative Phy cluster metadata.

    Individual ``cluster_<field>.tsv`` files are loaded and merged.
    ``cluster_info.tsv`` is ignored because Phy treats it as a derived
    snapshot rather than authoritative cluster metadata.
    """
    meta = pd.DataFrame(
        index=unit_ids,
    )

    paths = sorted(
        path for path in folder.glob('cluster_*.tsv') if path.name != 'cluster_info.tsv'
    )

    for path in paths:
        table = pd.read_csv(
            path,
            sep='\t',
        )

        if 'cluster_id' not in table.columns:
            raise ValueError(f'{path.name} does not contain cluster_id')

        cluster_id_column = table['cluster_id']

        if not isinstance(cluster_id_column, pd.Series):
            raise TypeError(f'{path.name} contains duplicate cluster_id columns')

        table = table.drop(
            columns=['cluster_id'],
        )

        cluster_ids = np.asarray(
            pd.to_numeric(
                cluster_id_column.to_numpy(),
                errors='raise',
            )
        )

        if not np.equal(
            cluster_ids,
            np.floor(cluster_ids),
        ).all():
            raise ValueError(f'{path.name} contains non-integer cluster IDs')

        table.index = pd.Index(
            cluster_ids.astype(np.int64),
            name='unit',
        )

        if not table.index.is_unique:
            raise ValueError(f'{path.name} contains duplicate cluster IDs')

        # spike_clusters.npy defines the current cluster set.
        # This deliberately drops stale metadata for clusters that
        # disappeared after Phy merges or splits.
        table = table.reindex(unit_ids)

        for column in table.columns:
            incoming = table[column]

            if not isinstance(incoming, pd.Series):
                raise TypeError(
                    f'{path.name} contains duplicate metadata column {column!r}'
                )

            if column not in meta.columns:
                meta[column] = incoming
                continue

            existing = meta[column]

            if not isinstance(existing, pd.Series):
                raise TypeError(f'duplicate metadata column {column!r}')

            overlap = existing.notna() & incoming.notna()
            same = existing.eq(incoming) | ~overlap

            if not same.fillna(False).all():
                logger.warning(
                    'Cluster metadata column %r disagrees in %s; '
                    'keeping earlier values where both are present',
                    column,
                    path.name,
                )

            meta[column] = existing.combine_first(incoming)

    return meta


def _group_spikes(
    times: np.ndarray,
    clusters: np.ndarray,
) -> tuple[pd.Index, list[np.ndarray]]:
    """Group globally ordered spike times by current cluster ID."""
    if len(times) != len(clusters):
        raise ValueError(
            'spike_times.npy and spike_clusters.npy '
            'must contain the same number of entries'
        )

    if len(times) == 0:
        return (
            pd.Index(
                [],
                dtype=np.int64,
                name='unit',
            ),
            [],
        )

    order = np.argsort(
        clusters,
        kind='stable',
    )

    sorted_clusters = clusters[order]
    sorted_times = times[order]

    unit_ids, starts = np.unique(
        sorted_clusters,
        return_index=True,
    )

    stops = np.r_[
        starts[1:],
        len(sorted_times),
    ]

    grouped = [
        sorted_times[start:stop]
        for start, stop in zip(
            starts,
            stops,
            strict=True,
        )
    ]

    return (
        pd.Index(
            unit_ids,
            dtype=np.int64,
            name='unit',
        ),
        grouped,
    )


def load_phy(
    folder: str | pathlib.Path,
    *,
    sampling_rate: float,
    sample_count: int,
) -> nocte._coll.trains.Trains:
    """
    Load Phy-compatible spike-sorting output as Trains.

    ``spike_clusters.npy`` defines the current unit assignment, including
    any manual merges or splits performed in Phy. Spike times are interpreted
    as sample indices and converted to milliseconds.

    Parameters
    ----------
    folder
        Phy/Kilosort output directory.
    sampling_rate
        Sampling rate of the source recording in Hz.
    sample_count
        Number of samples in the source recording. This defines the complete
        observation support, including periods containing no spikes.
    """
    folder = pathlib.Path(folder)

    sampling = nocte._core.sampling.SamplingRate(sampling_rate)

    if not isinstance(
        sample_count,
        (int, np.integer),
    ):
        raise TypeError('sample_count must be an integer')

    sample_count = int(sample_count)

    if sample_count < 0:
        raise ValueError('sample_count must be non-negative')

    spike_samples = _load_int_vector(folder / 'spike_times.npy')
    spike_clusters = _load_int_vector(folder / 'spike_clusters.npy')

    if len(spike_samples) != len(spike_clusters):
        raise ValueError(
            'spike_times.npy and spike_clusters.npy '
            'must contain the same number of entries'
        )

    if len(spike_samples) >= 2 and np.any(spike_samples[1:] < spike_samples[:-1]):
        raise ValueError('spike_times.npy must be monotonically non-decreasing')

    if len(spike_samples) and (
        spike_samples[0] < 0 or spike_samples[-1] >= sample_count
    ):
        raise ValueError('spike times lie outside the source recording')
    spike_times = sampling.samples_to_ms(spike_samples)

    unit_ids, times = _group_spikes(
        spike_times,
        spike_clusters,
    )

    meta = _load_cluster_meta(
        folder,
        unit_ids,
    )

    support = nocte._coll.windows.Win(
        0.0,
        sampling.samples_to_ms(sample_count),
    )

    return nocte._coll.trains.Trains.from_times(
        times,
        meta=meta,
        support=support,
    )
