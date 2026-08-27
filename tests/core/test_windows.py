# tests/core/test_windows.py

import logging

import numpy as np
import pandas as pd
import pytest

from nocte.core.windows import Win, Windows


def test_windows():
    wins = Windows.from_arrays(
        start=np.array([-100, -200]),
        stop=np.array([500, 300]),
        ref=np.array([1000, 2000]),
    )

    np.testing.assert_array_equal(wins.start, [-100.0, -200.0])
    np.testing.assert_array_equal(wins.stop, [500.0, 300.0])
    np.testing.assert_array_equal(wins.ref, [1000.0, 2000.0])

    assert len(wins) == 2
    assert len(wins.meta) == 2


def test_windows_geometry_is_float():
    wins = Windows.from_arrays(
        start=np.array([0, 1]),
        stop=np.array([10, 11]),
        ref=np.array([100, 200]),
    )

    assert np.issubdtype(wins.start.dtype, np.floating)
    assert np.issubdtype(wins.stop.dtype, np.floating)
    assert np.issubdtype(wins.ref.dtype, np.floating)


def test_windows_rejects_mismatched_geometry_lengths():
    with pytest.raises(ValueError):
        Windows.from_arrays(
            start=np.array([0, 1]),
            stop=np.array([10]),
            ref=np.array([100, 200]),
        )


def test_windows_rejects_mismatched_meta_length():
    meta = pd.DataFrame({'cat': ['a']})

    with pytest.raises(ValueError):
        Windows.from_arrays(
            start=np.array([0, 1]),
            stop=np.array([10, 11]),
            ref=np.array([100, 200]),
            meta=meta,
        )


@pytest.mark.parametrize(
    ('start', 'stop', 'ref'),
    [
        ([np.nan], [1], [0]),
        ([0], [np.inf], [0]),
        ([0], [1], [np.nan]),
    ],
)
def test_windows_rejects_nonfinite_geometry(start, stop, ref):
    with pytest.raises(ValueError):
        Windows.from_arrays(
            start=np.array(start),
            stop=np.array(stop),
            ref=np.array(ref),
        )


def test_windows_rejects_negative_length():
    with pytest.raises(ValueError):
        Windows.from_arrays(
            start=np.array([10]),
            stop=np.array([5]),
            ref=np.array([0]),
        )


def test_windows_rejects_geometry_columns_in_meta():
    meta = pd.DataFrame(
        {
            'cat': ['a'],
            'start': [123],
        }
    )

    with pytest.raises(ValueError):
        Windows.from_arrays(
            start=np.array([0]),
            stop=np.array([10]),
            ref=np.array([100]),
            meta=meta,
        )


def test_default_meta():
    wins = Windows.from_arrays(
        start=np.array([0, 0]),
        stop=np.array([10, 20]),
        ref=np.array([100, 200]),
    )

    assert wins.meta.empty
    assert len(wins.meta) == 2


def test_metadata_index_is_preserved():
    meta = pd.DataFrame(
        {'cat': ['a', 'b']},
        index=pd.Index([10, 20], name='event_id'),
    )

    wins = Windows.from_arrays(
        start=np.array([0, 0]),
        stop=np.array([10, 20]),
        ref=np.array([100, 200]),
        meta=meta,
    )

    pd.testing.assert_index_equal(wins.index, meta.index)


def test_length():
    wins = Windows.from_arrays(
        start=np.array([-100, 50]),
        stop=np.array([500, 250]),
        ref=np.array([1000, 2000]),
    )

    np.testing.assert_array_equal(
        wins.length,
        [600, 200],
    )


def test_time_at():
    wins = Windows.from_arrays(
        start=np.array([-100, -200]),
        stop=np.array([500, 200]),
        ref=np.array([1000, 2000]),
    )

    np.testing.assert_array_equal(
        wins.time_at('start'),
        [900, 1800],
    )
    np.testing.assert_array_equal(
        wins.time_at('ref'),
        [1000, 2000],
    )
    np.testing.assert_array_equal(
        wins.time_at('mid'),
        [1200, 2000],
    )
    np.testing.assert_array_equal(
        wins.time_at('stop'),
        [1500, 2200],
    )


def test_time_at_quantile():
    wins = Windows.from_arrays(
        start=np.array([0, -100]),
        stop=np.array([100, 300]),
        ref=np.array([1000, 2000]),
    )

    np.testing.assert_array_equal(
        wins.time_at(0.25),
        [1025, 2000],
    )


@pytest.mark.parametrize('q', [-0.1, 1.1])
def test_time_at_rejects_invalid_quantile(q):
    wins = Windows.from_arrays([0], [100])

    with pytest.raises(ValueError):
        wins.time_at(q)


def test_mid():
    wins = Windows.from_arrays(
        start=np.array([-100, -200]),
        stop=np.array([300, 200]),
        ref=np.array([1000, 2000]),
    )

    np.testing.assert_array_equal(
        wins.mid,
        [1100, 2000],
    )


def test_contains_is_half_open():
    wins = Windows.from_arrays(
        start=np.array([0, 0]),
        stop=np.array([100, 100]),
        ref=np.array([0, 100]),
    )

    np.testing.assert_array_equal(
        wins.contains(0),
        [True, False],
    )

    np.testing.assert_array_equal(
        wins.contains(100),
        [False, True],
    )

    np.testing.assert_array_equal(
        wins.contains(200),
        [False, False],
    )


def test_contained_in_accounts_for_ref():
    wins = Windows.from_arrays(
        start=np.array([0, -100, 0]),
        stop=np.array([100, 100, 200]),
        ref=np.array([1000, 1000, 1000]),
    )

    outer = Win(
        -100,
        100,
        ref=1000,
    )

    np.testing.assert_array_equal(
        wins.contained_in(outer),
        [True, True, False],
    )


def test_overlaps_is_half_open():
    wins = Windows.from_arrays(
        start=np.array([0, 100, 200]),
        stop=np.array([100, 200, 300]),
        ref=np.zeros(3),
    )

    other = Win(100, 200)

    np.testing.assert_array_equal(
        wins.overlaps(other),
        [False, True, False],
    )


def test_from_arrays_broadcasts_scalar_ref():
    wins = Windows.from_arrays(
        start=np.array([-100, -200]),
        stop=np.array([100, 200]),
        ref=1000,
    )

    np.testing.assert_array_equal(
        wins.ref,
        [1000, 1000],
    )


def test_from_arrays_broadcasts_scalar_geometry():
    wins = Windows.from_arrays(
        start=-100,
        stop=500,
        ref=np.array([1000, 2000, 3000]),
    )

    np.testing.assert_array_equal(
        wins.start,
        [-100, -100, -100],
    )
    np.testing.assert_array_equal(
        wins.stop,
        [500, 500, 500],
    )


def test_build_around():
    wins = Windows.build_around(
        [1000, 2000, 3000],
        Win(-100, 500),
    )

    np.testing.assert_array_equal(
        wins.start,
        [-100, -100, -100],
    )
    np.testing.assert_array_equal(
        wins.stop,
        [500, 500, 500],
    )
    np.testing.assert_array_equal(
        wins.ref,
        [1000, 2000, 3000],
    )


def test_build_around_includes_template_ref():
    wins = Windows.build_around(
        [1000, 2000],
        Win(-100, 500, ref=50),
    )

    np.testing.assert_array_equal(
        wins.ref,
        [1050, 2050],
    )


def test_build_around_preserves_series_index():
    marks = pd.Series(
        [1000, 2000],
        index=pd.Index([10, 20], name='event_id'),
    )

    wins = Windows.build_around(
        marks,
        Win(-100, 500),
    )

    pd.testing.assert_index_equal(
        wins.index,
        marks.index,
    )


def test_build_around_preserves_metadata():
    marks = pd.Series(
        [1000, 2000],
        index=pd.Index([10, 20], name='event_id'),
    )

    meta = pd.DataFrame(
        {'cat': ['a', 'b']},
        index=marks.index,
    )

    wins = Windows.build_around(
        marks,
        Win(-100, 500),
        meta=meta,
    )

    pd.testing.assert_frame_equal(
        wins.meta,
        meta,
    )


def test_build_centered():
    wins = Windows.build_centered(
        [1000, 2000],
        duration=400,
    )

    np.testing.assert_array_equal(
        wins.start,
        [-200, -200],
    )
    np.testing.assert_array_equal(
        wins.stop,
        [200, 200],
    )
    np.testing.assert_array_equal(
        wins.ref,
        [1000, 2000],
    )


def test_build_between():
    wins = Windows.build_between(
        np.array([100, 150, 230]),
    )

    np.testing.assert_array_equal(
        wins.start,
        [0, 0],
    )
    np.testing.assert_array_equal(
        wins.stop,
        [50, 80],
    )
    np.testing.assert_array_equal(
        wins.ref,
        [100, 150],
    )


def test_build_between_sorts_times():
    wins = Windows.build_between(
        np.array([230, 100, 150]),
    )

    np.testing.assert_array_equal(
        wins.ref,
        [100, 150],
    )
    np.testing.assert_array_equal(
        wins.stop,
        [50, 80],
    )


def test_build_between_preserves_boundary_provenance():
    times = pd.Series(
        [300, 100, 200],
        index=pd.Index(
            ['c', 'a', 'b'],
            name='event_id',
        ),
    )

    wins = Windows.build_between(times)

    assert list(wins.meta['start_event_id']) == [
        'a',
        'b',
    ]
    assert list(wins.meta['stop_event_id']) == [
        'b',
        'c',
    ]


def test_from_dict():
    wins = Windows.from_dict(
        {
            'pre': Win(-100, 0),
            'post': Win(0, 200),
        }
    )

    assert list(wins.meta['cat']) == [
        'pre',
        'post',
    ]

    np.testing.assert_array_equal(
        wins.start,
        [-100, 0],
    )
    np.testing.assert_array_equal(
        wins.stop,
        [0, 200],
    )


def test_get_uses_item_identity():
    meta = pd.DataFrame(
        index=pd.Index(
            [10, 20],
            name='event_id',
        )
    )

    wins = Windows.from_arrays(
        [-100, -200],
        [500, 300],
        [1000, 2000],
        meta=meta,
    )

    assert wins.get(20) == Win(
        -200,
        300,
        ref=2000,
    )


def test_items():
    meta = pd.DataFrame(
        index=pd.Index(
            ['a', 'b'],
            name='event_id',
        )
    )

    wins = Windows.from_arrays(
        [-100, -200],
        [500, 300],
        [1000, 2000],
        meta=meta,
    )

    assert list(wins.items()) == [
        (
            'a',
            Win(-100, 500, ref=1000),
        ),
        (
            'b',
            Win(-200, 300, ref=2000),
        ),
    ]


def test_around_preserves_metadata_and_identity():
    meta = pd.DataFrame(
        {'cat': ['a', 'b']},
        index=pd.Index(
            [10, 20],
            name='event_id',
        ),
    )

    wins = Windows.from_arrays(
        [-100, -100],
        [300, 300],
        [1000, 2000],
        meta=meta,
    )

    around = wins.around(
        Win(-50, 50),
        q='mid',
    )

    pd.testing.assert_frame_equal(
        around.meta,
        meta,
    )

    np.testing.assert_array_equal(
        around.ref,
        wins.mid,
    )

    np.testing.assert_array_equal(
        around.start,
        [-50, -50],
    )
    np.testing.assert_array_equal(
        around.stop,
        [50, 50],
    )


def test_before():
    wins = Windows.from_arrays(
        0,
        100,
        [1000, 2000],
    )

    before = wins.before(
        50,
        offset=20,
    )

    np.testing.assert_array_equal(
        before.start,
        [-70, -70],
    )
    np.testing.assert_array_equal(
        before.stop,
        [-20, -20],
    )
    np.testing.assert_array_equal(
        before.ref,
        [1000, 2000],
    )


def test_after():
    wins = Windows.from_arrays(
        0,
        100,
        [1000, 2000],
    )

    after = wins.after(
        50,
        offset=20,
    )

    np.testing.assert_array_equal(
        after.start,
        [20, 20],
    )
    np.testing.assert_array_equal(
        after.stop,
        [70, 70],
    )
    np.testing.assert_array_equal(
        after.ref,
        [1100, 2100],
    )


def test_change_preserves_ref():
    wins = Windows.from_arrays(
        [-100, -200],
        [500, 300],
        [1000, 2000],
    )

    changed = wins.change(
        pre=-50,
        post=100,
    )

    np.testing.assert_array_equal(
        changed.start,
        [-150, -250],
    )
    np.testing.assert_array_equal(
        changed.stop,
        [600, 400],
    )
    np.testing.assert_array_equal(
        changed.ref,
        wins.ref,
    )


def test_expand():
    wins = Windows.from_arrays(
        [0],
        [100],
        [1000],
    )

    expanded = wins.expand(20)

    np.testing.assert_array_equal(
        expanded.start,
        [-20],
    )
    np.testing.assert_array_equal(
        expanded.stop,
        [120],
    )


def test_shrink():
    wins = Windows.from_arrays(
        [0],
        [100],
        [1000],
    )

    shrunk = wins.shrink(20)

    np.testing.assert_array_equal(
        shrunk.start,
        [20],
    )
    np.testing.assert_array_equal(
        shrunk.stop,
        [80],
    )


def test_shift_changes_ref_only():
    wins = Windows.from_arrays(
        [-100, -200],
        [500, 300],
        [1000, 2000],
    )

    shifted = wins.shift(250)

    np.testing.assert_array_equal(
        shifted.start,
        wins.start,
    )
    np.testing.assert_array_equal(
        shifted.stop,
        wins.stop,
    )
    np.testing.assert_array_equal(
        shifted.ref,
        [1250, 2250],
    )


def test_shift_per_window():
    wins = Windows.from_arrays(
        0,
        100,
        [1000, 2000],
    )

    shifted = wins.shift(np.array([10, -20]))

    np.testing.assert_array_equal(
        shifted.ref,
        [1010, 1980],
    )


def test_shift_series_aligns_by_item_identity():
    meta = pd.DataFrame(
        index=pd.Index(
            ['a', 'b'],
            name='event_id',
        )
    )

    wins = Windows.from_arrays(
        0,
        100,
        [1000, 2000],
        meta=meta,
    )

    shifts = pd.Series(
        {
            'b': 20,
            'a': 10,
        }
    )

    shifted = wins.shift(shifts)

    np.testing.assert_array_equal(
        shifted.ref,
        [1010, 2020],
    )


def test_reanchor_preserves_realized_interval():
    wins = Windows.from_arrays(
        [-100, -200],
        [300, 400],
        [1000, 2000],
    )

    old_start = wins.time_at('start').copy()
    old_stop = wins.time_at('stop').copy()

    reanchored = wins.reanchor('mid')

    np.testing.assert_array_equal(
        reanchored.time_at('start'),
        old_start,
    )
    np.testing.assert_array_equal(
        reanchored.time_at('stop'),
        old_stop,
    )

    np.testing.assert_array_equal(
        reanchored.ref,
        wins.mid,
    )


def test_reanchor_mid_makes_geometry_centered():
    wins = Windows.from_arrays(
        [-100, -200],
        [300, 400],
        [1000, 2000],
    )

    reanchored = wins.reanchor('mid')

    np.testing.assert_array_equal(
        reanchored.start,
        -reanchored.stop,
    )


def test_crop_preserves_ref_and_metadata():
    meta = pd.DataFrame(
        {'cat': ['a', 'b']},
        index=pd.Index([10, 20]),
    )

    wins = Windows.from_arrays(
        start=[0, 0],
        stop=[200, 200],
        ref=[1000, 1300],
        meta=meta,
    )

    cropped = wins.crop(Win(0, 400, ref=1100))

    pd.testing.assert_frame_equal(
        cropped.meta,
        meta,
    )

    np.testing.assert_array_equal(
        cropped.ref,
        wins.ref,
    )

    np.testing.assert_array_equal(
        cropped.time_at('start'),
        [1100, 1300],
    )

    np.testing.assert_array_equal(
        cropped.time_at('stop'),
        [1200, 1500],
    )


def test_crop_disjoint_windows_become_empty():
    wins = Windows.from_arrays(
        start=0,
        stop=100,
        ref=[0, 1000],
    )

    cropped = wins.crop(Win(200, 300))

    np.testing.assert_array_equal(
        cropped.is_empty(),
        [True, True],
    )


def test_drop_empty():
    wins = Windows.from_arrays(
        start=[0, 10, 20],
        stop=[0, 20, 20],
        ref=[100, 200, 300],
        meta=pd.DataFrame({'cat': ['a', 'b', 'c']}),
    )

    result = wins.drop_empty()

    assert len(result) == 1
    assert result.meta.iloc[0]['cat'] == 'b'

    np.testing.assert_array_equal(
        result.ref,
        [200],
    )


def test_are_uniform():
    wins = Windows.from_arrays(
        start=-100,
        stop=500,
        ref=[1000, 2000, 3000],
    )

    assert wins.are_uniform()


def test_are_uniform_false():
    wins = Windows.from_arrays(
        start=[-100, -101],
        stop=[500, 500],
        ref=[1000, 2000],
    )

    assert not wins.are_uniform()


def test_are_exclusive_touching_windows():
    wins = Windows.from_arrays(
        start=0,
        stop=100,
        ref=[0, 100, 200],
    )

    assert wins.are_exclusive()


def test_are_exclusive_overlapping_windows():
    wins = Windows.from_arrays(
        start=0,
        stop=100,
        ref=[0, 50],
    )

    assert not wins.are_exclusive()


def test_are_exclusive_uses_realized_coordinates():
    wins = Windows.from_arrays(
        start=[-1000, 0],
        stop=[-900, 100],
        ref=[1000, 100],
    )

    # Realized intervals are [0, 100) and [100, 200).
    assert wins.are_exclusive()


def test_are_tight_touching_windows():
    wins = Windows.from_arrays(
        start=0,
        stop=100,
        ref=[0, 100, 200],
    )

    assert wins.are_tight()


def test_are_tight_with_gap():
    wins = Windows.from_arrays(
        start=0,
        stop=100,
        ref=[0, 101],
    )

    assert not wins.are_tight()


def test_are_tight_with_overlap():
    wins = Windows.from_arrays(
        start=0,
        stop=100,
        ref=[0, 50],
    )

    assert wins.are_tight()


def test_total():
    wins = Windows.from_arrays(
        start=[0, -100],
        stop=[100, 300],
        ref=[0, 1000],
    )

    assert wins.total() == 500.0


def test_global_win():
    wins = Windows.from_arrays(
        start=[-100, 0],
        stop=[200, 300],
        ref=[1000, 2000],
    )

    global_win = wins.global_win()

    assert global_win == Win(
        0,
        1400,
        ref=900,
    )


def test_global_win_rejects_empty_collection():
    wins = Windows.from_arrays(
        [],
        [],
        [],
    )

    with pytest.raises(ValueError):
        wins.global_win()


def test_sort_time_uses_realized_time():
    meta = pd.DataFrame({'name': ['late', 'early']})

    wins = Windows.from_arrays(
        start=[-1000, 0],
        stop=[-900, 100],
        ref=[2000, 100],
        meta=meta,
    )

    sorted_wins = wins.sort_time()

    assert list(sorted_wins.meta['name']) == [
        'early',
        'late',
    ]


def test_to_frame():
    meta = pd.DataFrame(
        {
            'cat': ['a', 'b'],
            'score': [1.5, 2.5],
        },
        index=pd.Index(
            [10, 20],
            name='event_id',
        ),
    )

    wins = Windows.from_arrays(
        start=[-100, -200],
        stop=[500, 300],
        ref=[1000, 2000],
        meta=meta,
    )

    df = wins.to_frame()

    expected = pd.DataFrame(
        {
            'start': [-100.0, -200.0],
            'stop': [500.0, 300.0],
            'ref': [1000.0, 2000.0],
            'cat': ['a', 'b'],
            'score': [1.5, 2.5],
        },
        index=meta.index,
    )

    pd.testing.assert_frame_equal(
        df,
        expected,
    )


def test_to_frame_is_a_copy():
    wins = Windows.from_arrays(
        [0],
        [100],
        [1000],
        meta=pd.DataFrame({'cat': ['a']}),
    )

    df = wins.to_frame()

    df.loc[0, 'cat'] = 'changed'
    df.loc[0, 'start'] = 999

    assert wins.meta.loc[0, 'cat'] == 'a'
    assert wins.start[0] == 0


def test_copy_has_independent_meta():
    wins = Windows.from_arrays(
        [0],
        [100],
        [1000],
        meta=pd.DataFrame({'cat': ['a']}),
    )

    copied = wins.copy()
    copied.meta.loc[0, 'cat'] = 'b'

    assert wins.meta.loc[0, 'cat'] == 'a'


def test_empty_window_makes_collection_nonexclusive_when_inside_another():
    wins = Windows.from_arrays(
        start=[0, 5],
        stop=[10, 5],
        ref=[0, 0],
    )

    assert not wins.are_exclusive()


def test_edges_include_empty_windows():
    wins = Windows.from_arrays(
        start=[0, 20],
        stop=[10, 20],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.edges(),
        [0, 10, 20],
    )


def test_global_win_includes_empty_windows():
    wins = Windows.from_arrays(
        start=[0, 100],
        stop=[10, 100],
        ref=0,
    )

    assert wins.global_win() == Win(
        0,
        100,
        ref=0,
    )


def test_edges():
    wins = Windows.from_arrays(
        start=[20, 0, 10],
        stop=[30, 10, 20],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.edges(),
        [0, 10, 20, 30],
    )


def test_edges_are_unique():
    wins = Windows.from_arrays(
        start=[0, 10],
        stop=[10, 20],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.edges(),
        [0, 10, 20],
    )


def test_edges_use_realized_coordinates():
    wins = Windows.from_arrays(
        start=[-100, -100],
        stop=[0, 0],
        ref=[1000, 2000],
    )

    np.testing.assert_array_equal(
        wins.edges(),
        [900, 1000, 1900, 2000],
    )


def test_breaks():
    wins = Windows.from_arrays(
        start=0,
        stop=10,
        ref=[0, 10, 20],
    )

    np.testing.assert_array_equal(
        wins.breaks(),
        [0, 10, 20, 30],
    )


def test_breaks_requires_exclusive_windows():
    wins = Windows.from_arrays(
        start=0,
        stop=10,
        ref=[0, 5],
    )

    with pytest.raises(ValueError):
        wins.breaks()


def test_breaks_requires_tight_windows():
    wins = Windows.from_arrays(
        start=0,
        stop=10,
        ref=[0, 20],
    )

    with pytest.raises(ValueError):
        wins.breaks()


def test_merge_overlap():
    wins = Windows.from_arrays(
        start=[0, 5, 20],
        stop=[10, 15, 30],
        ref=0,
    )

    merged = wins.merge_overlap()

    np.testing.assert_array_equal(
        merged.time_at('start'),
        [0, 20],
    )
    np.testing.assert_array_equal(
        merged.time_at('stop'),
        [15, 30],
    )


def test_merge_overlap_does_not_merge_touching():
    wins = Windows.from_arrays(
        start=[0, 10],
        stop=[10, 20],
        ref=0,
    )

    merged = wins.merge_overlap()

    assert len(merged) == 2


def test_merge_tight_merges_touching():
    wins = Windows.from_arrays(
        start=[0, 10, 25],
        stop=[10, 20, 30],
        ref=0,
    )

    merged = wins.merge_tight()

    np.testing.assert_array_equal(
        merged.time_at('start'),
        [0, 25],
    )
    np.testing.assert_array_equal(
        merged.time_at('stop'),
        [20, 30],
    )


def test_merge_overlap_records_provenance():
    wins = Windows.from_arrays(
        start=[0, 5, 20],
        stop=[10, 15, 30],
        ref=0,
    )

    merged = wins.merge_overlap()

    assert list(merged.meta['source_win_ids']) == [
        (0, 1),
        (2,),
    ]


def test_merge_overlap_take_first_metadata():
    meta = pd.DataFrame(
        {
            'cat': ['first', 'second'],
        }
    )

    wins = Windows.from_arrays(
        start=[0, 0],
        stop=[10, 10],
        ref=[0, 5],
        meta=meta,
    )

    merged = wins.merge_overlap(take='first')

    assert len(merged) == 1
    assert merged.meta.iloc[0]['cat'] == 'first'
    assert merged.ref[0] == 0


def test_merge_overlap_take_last_metadata_and_ref():
    meta = pd.DataFrame(
        {
            'cat': ['first', 'second'],
        }
    )

    wins = Windows.from_arrays(
        start=[0, 0],
        stop=[10, 10],
        ref=[0, 5],
        meta=meta,
    )

    merged = wins.merge_overlap(take='last')

    assert len(merged) == 1
    assert merged.meta.iloc[0]['cat'] == 'second'

    assert merged.ref[0] == 5

    assert merged.time_at('start')[0] == 0
    assert merged.time_at('stop')[0] == 15


def test_merge_overlap_rejects_invalid_take():
    wins = Windows.from_arrays(
        [0],
        [10],
        [0],
    )

    with pytest.raises(ValueError):
        wins.merge_overlap(
            take='banana',  # type: ignore
        )


def test_invert():
    wins = Windows.from_arrays(
        start=[2, 6],
        stop=[4, 8],
        ref=0,
    )

    inverted = wins.invert(Win(0, 10))

    np.testing.assert_array_equal(
        inverted.time_at('start'),
        [0, 4, 8],
    )
    np.testing.assert_array_equal(
        inverted.time_at('stop'),
        [2, 6, 10],
    )


def test_invert_uses_left_edge_as_ref():
    wins = Windows.from_arrays(
        start=[2],
        stop=[4],
        ref=0,
    )

    inverted = wins.invert(Win(0, 10))

    np.testing.assert_array_equal(
        inverted.start,
        [0, 0],
    )
    np.testing.assert_array_equal(
        inverted.ref,
        [0, 4],
    )
    np.testing.assert_array_equal(
        inverted.stop,
        [2, 6],
    )


def test_invert_merges_overlapping_coverage():
    wins = Windows.from_arrays(
        start=[2, 3],
        stop=[5, 7],
        ref=0,
    )

    inverted = wins.invert(Win(0, 10))

    np.testing.assert_array_equal(
        inverted.time_at('start'),
        [0, 7],
    )
    np.testing.assert_array_equal(
        inverted.time_at('stop'),
        [2, 10],
    )


def test_invert_empty_windows_do_not_remove_time():
    wins = Windows.from_arrays(
        start=[5],
        stop=[5],
        ref=0,
    )

    inverted = wins.invert(Win(0, 10))

    assert len(inverted) == 1
    assert inverted.get() == Win(
        0,
        10,
        ref=0,
    )


def test_invert_empty_collection():
    wins = Windows.from_arrays(
        [],
        [],
        [],
    )

    inverted = wins.invert(Win(0, 10))

    assert len(inverted) == 1
    assert inverted.get() == Win(
        0,
        10,
        ref=0,
    )


def test_split_left():
    wins = Windows.from_arrays(
        start=[0],
        stop=[10],
        ref=[100],
    )

    split = wins.split(
        3,
        align='left',
    )

    np.testing.assert_array_equal(
        split.time_at('start'),
        [100, 103, 106],
    )
    np.testing.assert_array_equal(
        split.time_at('stop'),
        [103, 106, 109],
    )


def test_split_right():
    wins = Windows.from_arrays(
        start=[0],
        stop=[10],
        ref=[100],
    )

    split = wins.split(
        3,
        align='right',
    )

    np.testing.assert_array_equal(
        split.time_at('start'),
        [101, 104, 107],
    )
    np.testing.assert_array_equal(
        split.time_at('stop'),
        [104, 107, 110],
    )


def test_split_numeric_alignment():
    wins = Windows.from_arrays(
        start=[0],
        stop=[10],
        ref=[100],
    )

    split = wins.split(
        3,
        align=0.5,
    )

    np.testing.assert_allclose(
        split.time_at('start'),
        [100.5, 103.5, 106.5],
    )
    np.testing.assert_allclose(
        split.time_at('stop'),
        [103.5, 106.5, 109.5],
    )


def test_split_preserves_source_ref():
    wins = Windows.from_arrays(
        [0],
        [10],
        [100],
    )

    split = wins.split(5)

    np.testing.assert_array_equal(
        split.ref,
        [100, 100],
    )


def test_split_duplicates_metadata_and_records_provenance():
    meta = pd.DataFrame(
        {
            'cat': ['rem'],
        },
        index=pd.Index(
            [17],
            name='win_id',
        ),
    )

    wins = Windows.from_arrays(
        [0],
        [10],
        [100],
        meta=meta,
    )

    split = wins.split(4)

    assert list(split.meta['cat']) == [
        'rem',
        'rem',
    ]

    assert list(split.meta['source_win_id']) == [
        17,
        17,
    ]

    assert list(split.meta['fragment_idx']) == [
        0,
        1,
    ]


def test_split_drops_remainder_smaller_than_fragment():
    wins = Windows.from_arrays(
        [0],
        [2],
        [0],
    )

    split = wins.split(3)

    assert len(split) == 0


@pytest.mark.parametrize(
    'length',
    [
        0,
        -1,
        np.inf,
        np.nan,
    ],
)
def test_split_rejects_invalid_length(length):
    wins = Windows.from_arrays(
        [0],
        [10],
        [0],
    )

    with pytest.raises(ValueError):
        wins.split(length)


@pytest.mark.parametrize(
    'align',
    [
        -0.1,
        1.1,
        'banana',
    ],
)
def test_split_rejects_invalid_alignment(align):
    wins = Windows.from_arrays(
        [0],
        [10],
        [0],
    )

    with pytest.raises(ValueError):
        wins.split(
            3,
            align=align,  # type: ignore
        )


def test_defrag():
    wins = Windows.from_arrays(
        start=[0, 0, 0],
        stop=[10, 20, 5],
        ref=[100, 300, 500],
    )

    defragged = wins.defrag()

    np.testing.assert_array_equal(
        defragged.time_at('start'),
        [0, 10, 30],
    )
    np.testing.assert_array_equal(
        defragged.time_at('stop'),
        [10, 30, 35],
    )


def test_defrag_preserves_relative_geometry():
    wins = Windows.from_arrays(
        start=[-2, -4],
        stop=[8, 16],
        ref=[100, 300],
    )

    defragged = wins.defrag()

    np.testing.assert_array_equal(
        defragged.start,
        wins.start,
    )
    np.testing.assert_array_equal(
        defragged.stop,
        wins.stop,
    )


def test_defrag_preserves_metadata_and_identity():
    meta = pd.DataFrame(
        {
            'cat': ['a', 'b'],
        },
        index=pd.Index(
            [10, 20],
            name='win_id',
        ),
    )

    wins = Windows.from_arrays(
        [0, 0],
        [10, 20],
        [100, 300],
        meta=meta,
    )

    defragged = wins.defrag()

    pd.testing.assert_frame_equal(
        defragged.meta,
        meta,
    )


def test_defrag_respects_current_item_order():
    wins = Windows.from_arrays(
        start=[0, 0],
        stop=[20, 10],
        ref=[200, 100],
    )

    defragged = wins.defrag()

    np.testing.assert_array_equal(
        defragged.time_at('start'),
        [0, 20],
    )


def test_defrag_custom_start():
    wins = Windows.from_arrays(
        [0, 0],
        [10, 20],
        [100, 300],
    )

    defragged = wins.defrag(
        start=500,
    )

    np.testing.assert_array_equal(
        defragged.time_at('start'),
        [500, 510],
    )


def test_interval_to_prev():
    wins = Windows.from_arrays(
        start=[20, 0, 12],
        stop=[25, 10, 18],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.interval_to_prev(),
        [2, np.inf, 2],
    )


def test_interval_to_next():
    wins = Windows.from_arrays(
        start=[20, 0, 12],
        stop=[25, 10, 18],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.interval_to_next(),
        [np.inf, 2, 2],
    )


def test_intervals_are_negative_for_overlap():
    wins = Windows.from_arrays(
        start=[0, 8],
        stop=[10, 20],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.interval_to_next(),
        [-2, np.inf],
    )

    np.testing.assert_array_equal(
        wins.interval_to_prev(),
        [np.inf, -2],
    )


def test_intervals_are_zero_for_touching():
    wins = Windows.from_arrays(
        start=[0, 10],
        stop=[10, 20],
        ref=0,
    )

    assert wins.interval_to_next()[0] == 0
    assert wins.interval_to_prev()[1] == 0


def test_interval_to_closest():
    wins = Windows.from_arrays(
        start=[20, 0, 12],
        stop=[25, 10, 18],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.interval_to_closest(),
        [2, 2, 2],
    )


def test_interval_shift():
    wins = Windows.from_arrays(
        start=[0, 10, 30],
        stop=[5, 15, 35],
        ref=0,
    )

    result = wins.interval_to_next(shift=2)

    np.testing.assert_array_equal(
        result,
        [25, np.inf, np.inf],
    )


@pytest.mark.parametrize(
    'method',
    [
        'interval_to_prev',
        'interval_to_next',
    ],
)
def test_interval_rejects_invalid_shift(method):
    wins = Windows.from_arrays(
        [0],
        [10],
        [0],
    )

    with pytest.raises(ValueError):
        getattr(wins, method)(0)


def test_is_isolated():
    wins = Windows.from_arrays(
        start=[0, 20, 50],
        stop=[10, 30, 60],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.is_isolated(10),
        [True, True, True],
    )


def test_is_isolated_asymmetric():
    wins = Windows.from_arrays(
        start=[0, 15, 40],
        stop=[10, 20, 50],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.is_isolated((5, 20)),
        [False, True, True],
    )


def test_classify_events_exclusive():
    wins = Windows.from_arrays(
        start=-10,
        stop=10,
        ref=[100, 200],
    )

    times = pd.Series(
        [90, 109, 110, 190, 209, 210],
        index=pd.Index(
            [10, 11, 12, 13, 14, 15],
            name='event_id',
        ),
    )

    classified = wins.classify_events(times)

    assert list(classified.index) == [
        10,
        11,
        13,
        14,
    ]

    assert list(classified['item_id']) == [
        0,
        0,
        1,
        1,
    ]

    np.testing.assert_array_equal(
        classified['delay'],
        [-10, 9, -10, 9],
    )


def test_classify_events_relative_to_start():
    wins = Windows.from_arrays(
        start=-10,
        stop=10,
        ref=[100],
    )

    classified = wins.classify_events(
        [95],
        relative_to='start',
    )

    assert classified.iloc[0]['delay'] == 5


def test_classify_events_relative_to_mid():
    wins = Windows.from_arrays(
        start=-10,
        stop=30,
        ref=[100],
    )

    classified = wins.classify_events(
        [115],
        relative_to='mid',
    )

    assert classified.iloc[0]['delay'] == 5


def test_classify_events_merges_metadata():
    meta = pd.DataFrame(
        {
            'cat': ['pre', 'post'],
        }
    )

    wins = Windows.from_arrays(
        start=0,
        stop=10,
        ref=[0, 10],
        meta=meta,
    )

    classified = wins.classify_events(
        [5, 15],
        merge_meta='cat',
    )

    assert list(classified['cat']) == [
        'pre',
        'post',
    ]


def test_classify_events_rejects_unknown_metadata():
    wins = Windows.from_arrays(
        [0],
        [10],
        [0],
    )

    with pytest.raises(KeyError):
        wins.classify_events(
            [5],
            merge_meta='does_not_exist',
        )


def test_classify_events_drops_events_outside_windows():
    wins = Windows.from_arrays(
        [0],
        [10],
        [100],
    )

    classified = wins.classify_events([50, 100, 105, 110, 200])

    np.testing.assert_array_equal(
        classified.index,
        [1, 2],
    )


def test_classify_events_overlapping_windows():
    wins = Windows.from_arrays(
        start=0,
        stop=10,
        ref=[0, 5],
    )

    times = pd.Series(
        [2, 7, 12],
        index=pd.Index(
            [100, 101, 102],
            name='event_id',
        ),
    )

    classified = wins.classify_events(times)

    assert list(classified.index) == [
        100,
        101,
        101,
        102,
    ]

    assert list(classified['item_id']) == [
        0,
        0,
        1,
        1,
    ]

    np.testing.assert_array_equal(
        classified['delay'],
        [2, 7, 2, 7],
    )


def test_classify_events_empty_input():
    wins = Windows.from_arrays(
        [0],
        [10],
        [0],
    )

    classified = wins.classify_events([])

    assert classified.empty


def test_windows_allows_empty_windows():
    wins = Windows.from_arrays(
        start=[0, 5],
        stop=[10, 5],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.is_empty(),
        [False, True],
    )


def test_empty_windows_warn(caplog):
    with caplog.at_level(
        logging.WARNING,
        logger='nocte.core.windows',
    ):
        Windows.from_arrays(
            start=[0, 5],
            stop=[10, 5],
            ref=0,
        )

    assert 'empty' in caplog.text.lower()


def test_empty_window_contains_no_time():
    wins = Windows.from_arrays(
        start=[5],
        stop=[5],
        ref=0,
    )

    np.testing.assert_array_equal(
        wins.contains(5),
        [False],
    )


@pytest.mark.parametrize(
    ('start', 'stop', 'ref'),
    [
        (0, 0, 0),
        (5, 5, 0),
        (10, 10, 0),
        (0, 0, 5),
    ],
)
def test_empty_window_does_not_overlap(start, stop, ref):
    wins = Windows.from_arrays(
        start=[start],
        stop=[stop],
        ref=[ref],
    )

    np.testing.assert_array_equal(
        wins.overlaps(Win(0, 10)),
        [False],
    )


def test_classify_events_empty_window_contains_no_events():
    wins = Windows.from_arrays(
        start=[5],
        stop=[5],
        ref=0,
    )

    classified = wins.classify_events([4, 5, 6])

    assert classified.empty


def test_merge_overlap_keeps_empty_window():
    wins = Windows.from_arrays(
        start=[0, 5],
        stop=[10, 5],
        ref=0,
    )

    merged = wins.merge_overlap()

    assert len(merged) == 2
