import numpy as np
import pandas as pd
import pytest

import nocte.core.frames
import nocte.core.grouping
import nocte.core.traces


def _make_traces() -> nocte.core.traces.Traces:
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )

    meta = pd.DataFrame(
        {
            'animal': ['a', 'a', 'b'],
            'side': ['left', 'right', 'left'],
            'condition': ['control', 'control', 'stim'],
        },
        index=pd.Index(
            [10, 20, 30],
            name='trace_id',
        ),
    )

    return nocte.core.traces.Traces.from_array(
        values,
        hz=1_000,
        meta=meta,
    )


def test_from_groupby():
    traces = _make_traces()

    grouping = nocte.core.grouping.Grouping.from_groupby(
        traces,
        by='animal',
        sort=True,
    )

    assert len(grouping) == 2
    assert grouping.index.name == 'group_id'
    assert grouping.index.tolist() == [0, 1]

    group_a = grouping.get(0)
    group_b = grouping.get(1)

    assert group_a.index.tolist() == [10, 20]
    assert group_b.index.tolist() == [30]

    assert grouping.meta.loc[0, 'animal'] == 'a'
    assert grouping.meta.loc[0, 'condition'] == 'control'

    assert grouping.meta.loc[1, 'animal'] == 'b'
    assert grouping.meta.loc[1, 'side'] == 'left'
    assert grouping.meta.loc[1, 'condition'] == 'stim'


def test_from_groupby_multiple_columns():
    traces = _make_traces()

    grouping = nocte.core.grouping.Grouping.from_groupby(
        traces,
        by=['animal', 'side'],
        sort=True,
    )

    assert len(grouping) == 3

    assert set(grouping.meta.columns) >= {
        'animal',
        'side',
        'condition',
    }


def test_grouping_selection_preserves_outer_ids():
    traces = _make_traces()

    grouping = nocte.core.grouping.Grouping.from_groupby(
        traces,
        by='animal',
        sort=True,
    )

    selected = grouping.sel_index([1])

    assert selected.index.tolist() == [1]
    assert len(selected) == 1

    assert selected.get(1).index.tolist() == [30]


def test_grouping_requires_homogeneous_concrete_type():
    traces = _make_traces()

    frames = nocte.core.frames.Frames.from_items([pd.DataFrame({'x': [1, 2]})])

    with pytest.raises(
        TypeError,
        match='same concrete type',
    ):
        nocte.core.grouping.Grouping.from_items([traces, frames])


def test_grouping_map():
    traces = _make_traces()

    grouping = nocte.core.grouping.Grouping.from_groupby(
        traces,
        by='animal',
        sort=True,
    )

    mapped = grouping.map(lambda group: group.sel_index([group.index[0]]))

    assert mapped.index.equals(grouping.index)

    pd.testing.assert_frame_equal(
        mapped.meta,
        grouping.meta,
    )

    assert mapped.get(0).index.tolist() == [10]
    assert mapped.get(1).index.tolist() == [30]


def test_grouping_apply():
    traces = _make_traces()

    grouping = nocte.core.grouping.Grouping.from_groupby(
        traces,
        by='animal',
        sort=True,
    )

    result = grouping.apply(len)

    expected = pd.Series(
        [2, 1],
        index=grouping.index,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_empty_grouping():
    grouping = nocte.core.grouping.Grouping.from_items([])

    assert len(grouping) == 0
    assert grouping.empty
    assert grouping.index.name == 'group_id'
