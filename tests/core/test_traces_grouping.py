import numpy as np
import pandas as pd
import pytest

from nocte._coll.traces import Traces, TracesGrouping
from nocte._core.sampling import TimeGrid


def _trace(
    values,
    *,
    hz,
    start,
    idx,
    name='trace',
    **meta,
):
    values = np.asarray(values, dtype=float).reshape(1, -1)

    df = pd.DataFrame(
        meta,
        index=pd.Index([idx], name=name),
    )

    return Traces.from_array(
        values,
        hz,
        start=start,
        meta=df,
    )


def _empty_trace(
    *,
    hz,
    start,
    idx,
    name='trace',
    **meta,
):
    return _trace(
        np.empty(0),
        hz=hz,
        start=start,
        idx=idx,
        name=name,
        **meta,
    )


def _grouping(
    traces,
    *,
    ids=None,
    name='win',
    **meta,
):
    if ids is None:
        ids = range(len(traces))

    group_meta = pd.DataFrame(
        meta,
        index=pd.Index(ids, name=name),
    )

    return TracesGrouping.from_items(
        traces,
        meta=group_meta,
    )


def test_time_grid_from_bounds():
    grid = TimeGrid.from_hz_bounds(
        hz=100,
        start=0,
        stop=25,
    )

    assert grid.start == 0
    assert grid.sampling.rate == 100
    assert grid.n_samples == 3
    assert grid.stop == 30

    np.testing.assert_allclose(
        grid.times,
        [0, 10, 20],
    )


def test_time_grid_empty_bounds():
    grid = TimeGrid.from_hz_bounds(
        hz=100,
        start=15,
        stop=15,
    )

    assert grid.start == 15
    assert grid.stop == 15
    assert grid.n_samples == 0
    assert grid.times.size == 0


def test_resample_to_grid_same_hz_different_phase():
    traces = _trace(
        [0, 1, 2],
        hz=100,
        start=0,
        idx=10,
    )

    grid = TimeGrid.from_hz_bounds(
        hz=100,
        start=5,
        stop=25,
    )

    result = traces.resample_to_grid(grid)

    assert result.start == 5
    assert result.hz == 100
    assert result.n_samples == 2

    np.testing.assert_allclose(
        result.time,
        [5, 15],
    )
    np.testing.assert_allclose(
        result.values,
        [[0.5, 1.5]],
    )


def test_grouping_resample_normalizes_different_grids():
    left = _trace(
        [0, 2, 4],
        hz=50,
        start=0,
        idx=10,
        name='beta',
    )

    right = _trace(
        [10, 20, 30, 40, 50],
        hz=100,
        start=10,
        idx=11,
        name='beta',
    )

    grouped = _grouping(
        [left, right],
        ids=[100, 200],
        name='win',
    )

    result = grouped.resample(
        200,
    )

    first = result.get(100)
    second = result.get(200)

    assert first.hz == second.hz == 200
    assert first.start == second.start == 0
    assert first.n_samples == second.n_samples == 12

    np.testing.assert_allclose(
        first.time,
        second.time,
    )
    np.testing.assert_allclose(
        first.time,
        np.arange(0, 60, 5),
    )

    np.testing.assert_allclose(
        first.values[0, :9],
        np.arange(0, 4.5, 0.5),
    )
    assert np.isnan(first.values[0, 9:]).all()

    assert np.isnan(second.values[0, :2]).all()

    np.testing.assert_allclose(
        second.values[0, 2:11],
        np.arange(10, 55, 5),
    )
    assert np.isnan(second.values[0, 11])


def test_concat_rejects_different_grids_then_accepts_resampled():
    left = _trace(
        [0, 1, 2],
        hz=50,
        start=0,
        idx=10,
        name='beta',
    )

    right = _trace(
        [3, 4, 5, 6],
        hz=100,
        start=10,
        idx=11,
        name='beta',
    )

    grouped = _grouping(
        [left, right],
        ids=[100, 200],
        name='win',
        condition=['a', 'b'],
    )

    with pytest.raises(
        ValueError,
        match='same time grid',
    ):
        grouped.concat()

    result = grouped.resample(
        200,
    ).concat()

    assert result.shape == (2, 12)
    assert result.hz == 200
    assert result.start == 0
    assert result.n_samples == 12

    pd.testing.assert_index_equal(
        result.index,
        pd.Index(
            [10, 11],
            name='beta',
        ),
    )

    pd.testing.assert_series_equal(
        result.meta['win'],
        pd.Series(
            [100, 200],
            index=result.index,
            name='win',
        ),
    )

    pd.testing.assert_series_equal(
        result.meta['condition'],
        pd.Series(
            ['a', 'b'],
            index=result.index,
            name='condition',
        ),
    )


def test_empty_trace_resamples_to_common_grid_and_concats():
    empty = _empty_trace(
        hz=100,
        start=10,
        idx=10,
        name='beta',
    )

    full = _trace(
        [1, 2, 3],
        hz=100,
        start=0,
        idx=11,
        name='beta',
    )

    grouped = _grouping(
        [empty, full],
        ids=[100, 200],
        name='win',
    )

    result = grouped.resample(
        100,
    ).concat()

    assert result.shape == (2, 3)
    assert result.start == 0
    assert result.stop == 30

    assert np.isnan(result.values[0]).all()

    np.testing.assert_allclose(
        result.values[1],
        [1, 2, 3],
    )


def test_zero_item_group_is_valid():
    empty_meta = pd.DataFrame(
        index=pd.Index(
            [],
            dtype=np.int64,
            name='beta',
        )
    )

    empty = Traces.from_array(
        np.empty((0, 3)),
        100,
        start=0,
        meta=empty_meta,
    )

    full = _trace(
        [1, 2, 3],
        hz=100,
        start=0,
        idx=10,
        name='beta',
    )

    grouped = _grouping(
        [empty, full],
        ids=[100, 200],
        name='win',
    )

    result = grouped.concat()

    assert result.shape == (1, 3)

    pd.testing.assert_index_equal(
        result.index,
        pd.Index(
            [10],
            name='beta',
        ),
    )

    assert result.meta.loc[10, 'win'] == 200


def test_all_zero_item_groups_can_concat():
    meta0 = pd.DataFrame(
        index=pd.Index(
            [],
            dtype=np.int64,
            name='beta',
        )
    )
    meta1 = meta0.copy()

    left = Traces.from_array(
        np.empty((0, 3)),
        100,
        start=0,
        meta=meta0,
    )
    right = Traces.from_array(
        np.empty((0, 3)),
        100,
        start=0,
        meta=meta1,
    )

    grouped = _grouping(
        [left, right],
        ids=[100, 200],
        name='win',
    )

    result = grouped.concat()

    assert result.shape == (0, 3)
    assert result.start == 0
    assert result.hz == 100
    assert result.index.empty
    assert result.index.name == 'beta'


def test_empty_grouping_cannot_infer_concat_grid():
    grouped = TracesGrouping.from_items(
        [],
        meta=pd.DataFrame(
            index=pd.Index(
                [],
                dtype=np.int64,
                name='win',
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match='empty TracesGrouping',
    ):
        grouped.concat()
