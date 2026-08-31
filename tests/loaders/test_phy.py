from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import nocte.core.windows
import nocte.loaders.phy


def _write_sorting(
    path,
    *,
    spike_times,
    spike_clusters,
) -> None:
    np.save(
        path / 'spike_times.npy',
        np.asarray(spike_times),
    )
    np.save(
        path / 'spike_clusters.npy',
        np.asarray(spike_clusters),
    )


def _write_cluster_meta(
    path,
    filename,
    data,
) -> None:
    pd.DataFrame(data).to_csv(
        path / filename,
        sep='\t',
        index=False,
    )


def test_load_phy(tmp_path):
    _write_sorting(
        tmp_path,
        spike_times=[[0], [10], [20], [30], [50]],
        spike_clusters=[[5], [5], [8], [5], [8]],
    )

    _write_cluster_meta(
        tmp_path,
        'cluster_group.tsv',
        {
            'cluster_id': [5, 8],
            'group': ['good', 'mua'],
        },
    )

    _write_cluster_meta(
        tmp_path,
        'cluster_KSLabel.tsv',
        {
            'cluster_id': [5, 8],
            'KSLabel': ['good', 'mua'],
        },
    )

    trains = nocte.loaders.phy.load_phy(
        tmp_path,
        sampling_rate=1000,
        sample_count=100,
    )

    assert trains.index.tolist() == [5, 8]
    assert trains.index.name == 'unit'

    pd.testing.assert_frame_equal(
        trains.meta,
        pd.DataFrame(
            {
                'KSLabel': ['good', 'mua'],
                'group': ['good', 'mua'],
            },
            index=pd.Index(
                [5, 8],
                name='unit',
            ),
        ),
    )

    np.testing.assert_array_equal(
        trains.get(5),
        np.array([0.0, 10.0, 30.0]),
    )
    np.testing.assert_array_equal(
        trains.get(8),
        np.array([20.0, 50.0]),
    )

    assert trains.support == nocte.core.windows.Win(0, 100)


def test_load_phy_uses_current_clusters_and_drops_stale_metadata(tmp_path):
    _write_sorting(
        tmp_path,
        spike_times=[0, 10, 20, 30],
        spike_clusters=[5, 5, 8, 8],
    )

    _write_cluster_meta(
        tmp_path,
        'cluster_group.tsv',
        {
            'cluster_id': [1, 2, 5, 8],
            'group': ['noise', 'good', 'good', 'mua'],
        },
    )

    trains = nocte.loaders.phy.load_phy(
        tmp_path,
        sampling_rate=1000,
        sample_count=100,
    )

    assert trains.index.tolist() == [5, 8]
    assert trains.meta['group'].tolist() == ['good', 'mua']


def test_load_phy_ignores_cluster_info(tmp_path):
    _write_sorting(
        tmp_path,
        spike_times=[0, 10, 20, 30],
        spike_clusters=[5, 5, 8, 8],
    )

    _write_cluster_meta(
        tmp_path,
        'cluster_group.tsv',
        {
            'cluster_id': [5, 8],
            'group': ['good', 'mua'],
        },
    )

    _write_cluster_meta(
        tmp_path,
        'cluster_info.tsv',
        {
            'cluster_id': [5, 8],
            'group': ['noise', 'noise'],
            'n_spikes': [123, 456],
        },
    )

    trains = nocte.loaders.phy.load_phy(
        tmp_path,
        sampling_rate=1000,
        sample_count=100,
    )

    assert trains.meta['group'].tolist() == ['good', 'mua']
    assert 'n_spikes' not in trains.meta.columns


def test_load_phy_accepts_vector_and_column_arrays(tmp_path):
    _write_sorting(
        tmp_path,
        spike_times=np.array([[0], [10], [20]]),
        spike_clusters=np.array([2, 2, 4]),
    )

    trains = nocte.loaders.phy.load_phy(
        tmp_path,
        sampling_rate=1000,
        sample_count=100,
    )

    assert trains.index.tolist() == [2, 4]

    np.testing.assert_array_equal(
        trains.get(2),
        np.array([0.0, 10.0]),
    )
    np.testing.assert_array_equal(
        trains.get(4),
        np.array([20.0]),
    )


def test_load_phy_converts_samples_to_milliseconds(tmp_path):
    _write_sorting(
        tmp_path,
        spike_times=[0, 15_000, 30_000],
        spike_clusters=[0, 0, 0],
    )

    trains = nocte.loaders.phy.load_phy(
        tmp_path,
        sampling_rate=30_000,
        sample_count=60_000,
    )

    np.testing.assert_array_equal(
        trains.get(0),
        np.array([0.0, 500.0, 1000.0]),
    )

    assert trains.support == nocte.core.windows.Win(
        0,
        2000,
    )


def test_load_phy_rejects_mismatched_spike_arrays(tmp_path):
    _write_sorting(
        tmp_path,
        spike_times=[0, 10, 20],
        spike_clusters=[0, 0],
    )

    with pytest.raises(
        ValueError,
        match='same number of entries',
    ):
        nocte.loaders.phy.load_phy(
            tmp_path,
            sampling_rate=1000,
            sample_count=100,
        )


def test_load_phy_rejects_unsorted_spike_times(tmp_path):
    _write_sorting(
        tmp_path,
        spike_times=[0, 20, 10],
        spike_clusters=[0, 0, 0],
    )

    with pytest.raises(
        ValueError,
        match='monotonically non-decreasing',
    ):
        nocte.loaders.phy.load_phy(
            tmp_path,
            sampling_rate=1000,
            sample_count=100,
        )


@pytest.mark.parametrize(
    'spike_times',
    [
        [-1, 10],
        [0, 100],
    ],
)
def test_load_phy_rejects_spikes_outside_recording(
    tmp_path,
    spike_times,
):
    _write_sorting(
        tmp_path,
        spike_times=spike_times,
        spike_clusters=[0, 0],
    )

    with pytest.raises(
        ValueError,
        match='outside the source recording',
    ):
        nocte.loaders.phy.load_phy(
            tmp_path,
            sampling_rate=1000,
            sample_count=100,
        )
