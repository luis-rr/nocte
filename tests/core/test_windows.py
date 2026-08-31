import logging

import numpy as np
import pandas as pd
import pytest

from nocte.core.windows import Win, WindowMatches, Windows


def assert_series_values(series: pd.Series, values) -> None:
    np.testing.assert_array_equal(series.to_numpy(), np.asarray(values))


# -----------------------------------------------------------------------------
# core collection and access


def test_windows_preserve_collection_invariants_and_public_index():
    meta = pd.DataFrame(
        {'state': ['a', 'b']},
        index=pd.Index([10, 20], name='win_id'),
    )

    wins = Windows.from_arrays(
        start=[-1, -2],
        stop=[3, 4],
        ref=[100, 200],
        meta=meta,
    )

    assert len(wins) == len(wins.meta) == 2
    assert wins.index.equals(meta.index)

    for series in (wins.start, wins.stop, wins.ref, wins.lengths, wins.mid):
        assert isinstance(series, pd.Series)
        assert series.index.equals(meta.index)

    assert_series_values(wins.start, [-1.0, -2.0])
    assert_series_values(wins.stop, [3.0, 4.0])
    assert_series_values(wins.ref, [100.0, 200.0])
    assert_series_values(wins.lengths, [4.0, 6.0])


def test_from_arrays_broadcasts_scalars_but_not_length_one_arrays():
    wins = Windows.from_arrays(
        start=0,
        stop=[10, 20],
        ref=100,
    )

    assert_series_values(wins.start, [0, 0])
    assert_series_values(wins.ref, [100, 100])

    with pytest.raises(ValueError):
        Windows.from_arrays(
            start=[0],
            stop=[10, 20],
            ref=0,
        )


def test_windows_reject_invalid_geometry():
    with pytest.raises(ValueError):
        Windows.from_arrays([0], [-1])

    with pytest.raises(ValueError):
        Windows.from_arrays([0], [np.inf])

    with pytest.raises(ValueError):
        Windows.from_arrays(
            [0],
            [1],
            meta=pd.DataFrame(index=[0, 1]),
        )


def test_to_frame_allows_geometry_metadata_collisions_with_warning():
    wins = Windows.from_arrays(
        [0, 1],
        [10, 11],
        ref=[100, 200],
        meta=pd.DataFrame(
            {'start': [123, 456], 'state': ['a', 'b']},
            index=pd.Index([0, 1], name='win'),
        ),
    )

    with pytest.warns(UserWarning, match='duplicate columns'):
        frame = wins.to_frame()

    assert frame.columns.tolist() == ['start', 'stop', 'ref', 'start', 'state']
    np.testing.assert_array_equal(frame.iloc[:, 0], [0, 1])
    np.testing.assert_array_equal(frame.iloc[:, 3], [123, 456])


def test_empty_windows_are_valid_and_warn(caplog):
    with caplog.at_level(logging.WARNING, logger='nocte.core.windows'):
        wins = Windows.from_arrays(
            start=[0, 5],
            stop=[10, 5],
        )

    assert_series_values(wins.is_empty(), [False, True])
    assert '1/2 windows are empty' in caplog.text


def test_copy_shares_geometry_but_copies_metadata_semantically():
    wins = Windows.from_arrays(
        [0, 10],
        [10, 20],
        meta=pd.DataFrame({'state': ['a', 'b']}, index=pd.Index([0, 1], name='win')),
    )

    copied = wins.copy()
    copied.meta.loc[copied.index[0], 'state'] = 'changed'

    assert wins.meta.iloc[0]['state'] == 'a'
    assert_series_values(copied.start, wins.start)


def test_get_uses_item_identity_and_items_preserve_it():
    meta = pd.DataFrame(index=pd.Index([10, 20], name='win_id'))
    wins = Windows.from_arrays(
        [0, 0],
        [10, 20],
        [100, 200],
        meta=meta,
    )

    assert wins.get(20) == Win(0, 20, ref=200)
    assert list(wins.items()) == [
        (10, Win(0, 10, ref=100)),
        (20, Win(0, 20, ref=200)),
    ]

    with pytest.raises(ValueError):
        wins.get()

    assert Windows.from_arrays(0, 10, 100).get() == Win(0, 10, ref=100)


# -----------------------------------------------------------------------------
# constructors


def test_build_around_preserves_series_identity_and_template_reference():
    marks = pd.Series(
        [100, 200],
        index=pd.Index([10, 20], name='event_id'),
    )

    wins = Windows.build_around(
        marks,
        Win(-5, 10, ref=2),
    )

    assert wins.index.equals(marks.index)
    assert_series_values(wins.start, [-5, -5])
    assert_series_values(wins.stop, [10, 10])
    assert_series_values(wins.ref, [102, 202])


def test_build_centered():
    wins = Windows.build_centered([100, 200], 20)

    assert_series_values(wins.start, [-10, -10])
    assert_series_values(wins.stop, [10, 10])
    assert_series_values(wins.ref, [100, 200])


def test_build_between_sorts_times_and_records_boundary_provenance():
    times = pd.Series(
        [20, 0, 10],
        index=pd.Index([300, 100, 200], name='event_id'),
    )

    wins = Windows.build_between(times)

    assert_series_values(wins.ref, [0, 10])
    assert_series_values(wins.start, [0, 0])
    assert_series_values(wins.stop, [10, 10])
    assert list(wins.meta['start_event_id']) == [100, 200]
    assert list(wins.meta['stop_event_id']) == [200, 300]


def test_from_dict_keeps_labels_as_metadata():
    wins = Windows.from_dict(
        {
            'pre': (-10, 0),
            'post': Win(0, 20, ref=5),
        },
        name='phase',
    )

    assert list(wins.meta['phase']) == ['pre', 'post']
    assert_series_values(wins.ref, [0, 5])


def test_from_contiguous_values_builds_half_sample_boundaries():
    values = pd.Series(
        ['a', 'a', 'b', 'b'],
        index=pd.Index([5.0, 15.0, 25.0, 35.0], name='time'),
        name='state',
    )

    wins = Windows.from_contiguous_values(values)

    assert list(wins.meta['state']) == ['a', 'b']
    np.testing.assert_array_equal(wins.time_at('start'), [0, 20])
    np.testing.assert_array_equal(wins.time_at('stop'), [20, 40])


def test_contiguous_values_round_trip_exactly():
    values = pd.Series(
        ['a', 'a', 'b', 'b', 'a'],
        index=pd.Index([5.0, 15.0, 25.0, 35.0, 45.0], name='time'),
        name='state',
    )

    restored = Windows.from_contiguous_values(values).generate_contiguous(
        10,
        by='state',
    )

    pd.testing.assert_series_equal(restored, values)


def test_from_contiguous_values_rejects_missing_or_irregular_samples():
    with pytest.raises(ValueError):
        Windows.from_contiguous_values(pd.Series(['a', None], index=[0.0, 10.0]))

    with pytest.raises(ValueError):
        Windows.from_contiguous_values(
            pd.Series(['a', 'b', 'c'], index=[0.0, 10.0, 21.0])
        )


def test_from_contiguous_values_single_sample_requires_step():
    values = pd.Series(['a'], index=[10.0], name='state')

    with pytest.raises(ValueError):
        Windows.from_contiguous_values(values)

    wins = Windows.from_contiguous_values(values, step=4)
    assert wins.get() == Win(0, 4, ref=8)


# -----------------------------------------------------------------------------
# per-window geometry


def test_time_at_and_geometry_masks_preserve_item_indices():
    meta = pd.DataFrame(index=pd.Index([10, 20], name='win_id'))
    wins = Windows.from_arrays(
        start=[0, 0],
        stop=[10, 10],
        ref=[0, 10],
        meta=meta,
    )

    pd.testing.assert_series_equal(
        wins.time_at('mid'),
        pd.Series([5.0, 15.0], index=meta.index, name='mid'),
    )
    pd.testing.assert_series_equal(
        wins.contains(10),
        pd.Series([False, True], index=meta.index, name='contains'),
    )
    pd.testing.assert_series_equal(
        wins.is_empty(),
        pd.Series([False, False], index=meta.index, name='is_empty'),
    )


def test_contained_in_and_overlaps_use_realized_coordinates():
    wins = Windows.from_arrays(
        start=[-5, 0, 10],
        stop=[5, 10, 20],
        ref=[100, 100, 100],
    )

    assert_series_values(
        wins.contained_in(Win(95, 110)),
        [True, True, False],
    )
    assert_series_values(
        wins.overlaps(Win(105, 115)),
        [False, True, True],
    )


# -----------------------------------------------------------------------------
# per-window transformations


def test_around_preserves_metadata_and_identity():
    meta = pd.DataFrame(
        {'state': ['a', 'b']},
        index=pd.Index([10, 20], name='win_id'),
    )
    wins = Windows.from_arrays(0, 10, [100, 200], meta=meta)

    result = wins.around((-2, 3), q='start')

    assert result.index.equals(wins.index)
    pd.testing.assert_frame_equal(result.meta, wins.meta)
    assert_series_values(result.start, [-2, -2])
    assert_series_values(result.stop, [3, 3])
    assert_series_values(result.ref, [100, 200])


def test_centered_before_and_after_share_win_semantics():
    wins = Windows.from_arrays(0, 20, [100, 200])

    centered = wins.centered(10, q='ref')
    before = wins.before(10, offset=2, q='mid')
    after = wins.after(10, offset=2, q='mid')

    assert_series_values(centered.start, [-5, -5])
    assert_series_values(centered.stop, [5, 5])
    np.testing.assert_array_equal(before.time_at('start'), [98, 198])
    np.testing.assert_array_equal(before.time_at('stop'), [108, 208])
    np.testing.assert_array_equal(after.time_at('start'), [112, 212])
    np.testing.assert_array_equal(after.time_at('stop'), [122, 222])


def test_change_aligns_series_values_by_item_identity():
    meta = pd.DataFrame(index=pd.Index([10, 20], name='win_id'))
    wins = Windows.from_arrays([0, 0], [10, 10], meta=meta)
    pre = pd.Series([2, 1], index=[20, 10])

    changed = wins.change(pre=pre, post=[1, 2])

    assert_series_values(changed.start, [1, 2])
    assert_series_values(changed.stop, [11, 12])
    assert changed.index.equals(wins.index)


def test_shrink_expand_and_shift_support_one_value_per_window():
    meta = pd.DataFrame(index=pd.Index([10, 20], name='win_id'))
    wins = Windows.from_arrays([0, 0], [10, 20], [100, 200], meta=meta)

    shrunk = wins.shrink([1, 2])
    expanded = wins.expand([1, 2])
    shifted = wins.shift(pd.Series([20, 10], index=[20, 10]))

    assert_series_values(shrunk.start, [1, 2])
    assert_series_values(shrunk.stop, [9, 18])
    assert_series_values(expanded.start, [-1, -2])
    assert_series_values(expanded.stop, [11, 22])
    assert_series_values(shifted.ref, [110, 220])
    assert_series_values(shifted.start, wins.start)
    assert shifted.index.equals(wins.index)


@pytest.mark.parametrize(
    'index',
    [[20], [10, 20, 30], [10, 10]],
)
def test_per_window_series_rejects_nonmatching_identities(index):
    wins = Windows.from_arrays(
        [0, 0],
        [10, 20],
        meta=pd.DataFrame(index=pd.Index([10, 20], name='win_id')),
    )

    with pytest.raises(ValueError, match='exactly the collection index'):
        wins.shift(pd.Series([1] * len(index), index=index))


@pytest.mark.parametrize('method', ['shrink', 'expand'])
def test_resize_rejects_negative_duration(method):
    wins = Windows.from_arrays([0], [10])

    with pytest.raises(ValueError):
        getattr(wins, method)(-1)


def test_reanchor_preserves_realized_intervals():
    wins = Windows.from_arrays(
        start=[-10, 0],
        stop=[20, 30],
        ref=[100, 200],
    )
    before_start = wins.time_at('start').copy()
    before_stop = wins.time_at('stop').copy()

    result = wins.reanchor('mid')

    np.testing.assert_array_equal(result.time_at('start'), before_start)
    np.testing.assert_array_equal(result.time_at('stop'), before_stop)
    assert_series_values(result.start, [-15, -15])
    assert_series_values(result.stop, [15, 15])


def test_crop_preserves_identity_reference_and_turns_disjoint_windows_empty():
    meta = pd.DataFrame(index=pd.Index([10, 20], name='win_id'))
    wins = Windows.from_arrays(0, 10, [100, 200], meta=meta)

    cropped = wins.crop(Win(105, 115))

    assert cropped.index.equals(wins.index)
    assert_series_values(cropped.ref, [100, 200])
    np.testing.assert_array_equal(cropped.time_at('start'), [105, 115])
    np.testing.assert_array_equal(cropped.time_at('stop'), [110, 115])
    assert_series_values(cropped.is_empty(), [False, True])


def test_drop_empty_preserves_surviving_identity():
    meta = pd.DataFrame(index=pd.Index([10, 20, 30], name='win_id'))
    wins = Windows.from_arrays(
        [0, 5, 10],
        [1, 5, 11],
        meta=meta,
    )

    result = wins.drop_empty()

    assert list(result.index) == [10, 30]
    np.testing.assert_array_equal(result.time_at('start'), [0, 10])


# -----------------------------------------------------------------------------
# collection geometry


def test_are_uniform_uses_absolute_tolerance_only():
    assert Windows.from_arrays(
        [0, 0],
        [10, 10],
        [0, 100],
    ).are_uniform()

    wins = Windows.from_arrays(
        [1_000_000.0, 1_000_001.0],
        [1_000_010.0, 1_000_011.0],
    )
    assert not wins.are_uniform(atol=0.1)


def test_exclusive_and_tight_have_coverage_semantics():
    touching = Windows.from_arrays([0, 10], [10, 20])
    overlapping = Windows.from_arrays([0, 5], [10, 15])
    gapped = Windows.from_arrays([0, 11], [10, 20])
    with_empty = Windows.from_arrays([0, 5], [10, 5])

    assert touching.are_exclusive()
    assert touching.are_tight()
    assert not overlapping.are_exclusive()
    assert overlapping.are_tight()
    assert gapped.are_exclusive()
    assert not gapped.are_tight()
    assert with_empty.are_exclusive()
    assert with_empty.are_tight()


def test_bounding_win_uses_nonempty_coverage_only():
    wins = Windows.from_arrays(
        start=[0, 100],
        stop=[10, 100],
    )

    assert wins.bounding_win() == Win(0, 10, ref=0)

    with pytest.raises(ValueError):
        Windows.from_arrays([5], [5]).bounding_win()


def test_sort_time_uses_realized_coordinates_and_preserves_ids():
    meta = pd.DataFrame(index=pd.Index([10, 20, 30], name='win_id'))
    wins = Windows.from_arrays(
        start=[0, 0, 0],
        stop=[5, 5, 5],
        ref=[20, 0, 10],
        meta=meta,
    )

    result = wins.sort_time()

    assert list(result.index) == [20, 30, 10]


def test_edges_include_empty_items_but_breaks_use_coverage_only():
    wins = Windows.from_arrays(
        start=[0, 10, 15],
        stop=[10, 20, 15],
    )

    np.testing.assert_array_equal(wins.edges(), [0, 10, 15, 20])
    np.testing.assert_array_equal(wins.breaks(), [0, 10, 20])


def test_breaks_requires_tight_exclusive_coverage():
    with pytest.raises(ValueError):
        Windows.from_arrays([0, 11], [10, 20]).breaks()

    with pytest.raises(ValueError):
        Windows.from_arrays([0, 5], [10, 15]).breaks()


# -----------------------------------------------------------------------------
# structural transformations


def test_merge_overlap_and_merge_tight_distinguish_overlap_from_touching():
    wins = Windows.from_arrays(
        start=[0, 5, 15],
        stop=[10, 15, 20],
    )

    overlap = wins.merge_overlap()
    tight = wins.merge_tight()

    np.testing.assert_array_equal(overlap.time_at('start'), [0, 15])
    np.testing.assert_array_equal(overlap.time_at('stop'), [15, 20])
    np.testing.assert_array_equal(tight.time_at('start'), [0])
    np.testing.assert_array_equal(tight.time_at('stop'), [20])


def test_merge_by_requires_same_metadata_value():
    meta = pd.DataFrame(
        {'state': ['a', 'b', 'b']}, index=pd.Index([0, 1, 2], name='win')
    )
    wins = Windows.from_arrays(
        start=[0, 10, 20],
        stop=[10, 20, 30],
        meta=meta,
    )

    result = wins.merge_tight(by='state')

    assert list(result.meta['state']) == ['a', 'b']
    np.testing.assert_array_equal(result.time_at('start'), [0, 10])
    np.testing.assert_array_equal(result.time_at('stop'), [10, 30])


def test_merge_records_provenance_and_take_controls_metadata_and_reference():
    meta = pd.DataFrame(
        {'state': ['first', 'last']},
        index=pd.Index([10, 20], name='win_id'),
    )
    wins = Windows.from_arrays(
        start=[0, 0],
        stop=[10, 10],
        ref=[0, 5],
        meta=meta,
    )

    result = wins.merge_overlap(take='last')

    assert len(result) == 1
    assert result.meta.iloc[0]['state'] == 'last'
    assert result.meta.iloc[0]['source_win_ids'] == (10, 20)
    assert result.ref.iloc[0] == 5
    assert result.time_at('start').iloc[0] == 0
    assert result.time_at('stop').iloc[0] == 15


def test_merge_keeps_empty_windows_without_interrupting_coverage_merge():
    wins = Windows.from_arrays(
        start=[0, 5, 6],
        stop=[10, 5, 12],
    )

    result = wins.merge_overlap()

    assert len(result) == 2
    np.testing.assert_array_equal(result.time_at('start'), [0, 5])
    np.testing.assert_array_equal(result.time_at('stop'), [12, 5])
    assert list(result.meta['source_win_ids']) == [(0, 2), (1,)]


def test_invert_returns_uncovered_regions_and_ignores_empty_coverage():
    wins = Windows.from_arrays(
        start=[2, 4, 8],
        stop=[5, 7, 8],
    )

    result = wins.invert(Win(0, 10))

    np.testing.assert_array_equal(result.time_at('start'), [0, 7])
    np.testing.assert_array_equal(result.time_at('stop'), [2, 10])


def test_invert_empty_coverage_returns_requested_window():
    result = Windows.from_arrays([5], [5]).invert(Win(0, 10))

    assert result.get() == Win(0, 10, ref=0)


@pytest.mark.parametrize(
    ('align', 'starts'),
    [
        ('left', [100, 104]),
        ('right', [102, 106]),
        (0.5, [101, 105]),
    ],
)
def test_split_alignment(align, starts):
    wins = Windows.from_arrays(0, 10, 100)

    result = wins.split(4, align=align)

    np.testing.assert_array_equal(result.time_at('start'), starts)
    np.testing.assert_array_equal(result.time_at('stop'), np.asarray(starts) + 4)


def test_split_records_provenance_and_copies_metadata():
    meta = pd.DataFrame(
        {'state': ['rem']},
        index=pd.Index([17], name='win_id'),
    )
    wins = Windows.from_arrays(0, 10, 100, meta=meta)

    result = wins.split(4)

    assert list(result.meta['source_win_id']) == [17, 17]
    assert list(result.meta['fragment_idx']) == [0, 1]
    assert list(result.meta['state']) == ['rem', 'rem']


def test_split_handles_floating_point_exact_multiples():
    result = Windows.from_arrays(0, 0.3).split(0.1)

    assert len(result) == 3
    np.testing.assert_allclose(result.lengths, [0.1, 0.1, 0.1])


def test_defrag_preserves_order_identity_and_durations():
    meta = pd.DataFrame(index=pd.Index([10, 20], name='win_id'))
    wins = Windows.from_arrays(
        start=[0, 0],
        stop=[20, 10],
        ref=[200, 100],
        meta=meta,
    )

    result = wins.defrag(start=50)

    assert result.index.equals(wins.index)
    assert_series_values(result.lengths, [20, 10])
    np.testing.assert_array_equal(result.time_at('start'), [50, 70])
    np.testing.assert_array_equal(result.time_at('stop'), [70, 80])


# -----------------------------------------------------------------------------
# temporal relationships


def test_neighbor_intervals_are_chronological_but_return_in_item_order():
    meta = pd.DataFrame(index=pd.Index([30, 10, 20], name='win_id'))
    wins = Windows.from_arrays(
        start=[20, 0, 12],
        stop=[25, 10, 18],
        meta=meta,
    )

    pd.testing.assert_series_equal(
        wins.interval_to_prev(),
        pd.Series([2.0, np.inf, 2.0], index=meta.index, name='interval_to_prev'),
    )
    pd.testing.assert_series_equal(
        wins.interval_to_next(),
        pd.Series([np.inf, 2.0, 2.0], index=meta.index, name='interval_to_next'),
    )
    pd.testing.assert_series_equal(
        wins.interval_to_closest(),
        pd.Series([2.0, 2.0, 2.0], index=meta.index, name='interval_to_closest'),
    )


def test_neighbor_intervals_support_nth_neighbor():
    wins = Windows.from_arrays(
        start=[0, 10, 30],
        stop=[5, 15, 35],
    )

    assert_series_values(wins.interval_to_next(n=2), [25, np.inf, np.inf])
    assert_series_values(wins.interval_to_prev(n=2), [np.inf, np.inf, 25])


def test_is_isolated_supports_asymmetric_thresholds():
    wins = Windows.from_arrays(
        start=[0, 20, 50],
        stop=[10, 30, 60],
    )

    assert_series_values(wins.is_isolated((10, 20)), [False, True, True])


def test_is_isolated_rejects_overlapping_windows():
    wins = Windows.from_arrays([0, 5], [10, 15])

    with pytest.raises(ValueError):
        wins.is_isolated(1)


# -----------------------------------------------------------------------------
# event matching and categorical operations


def test_match_events_returns_sparse_relation_for_overlapping_windows():
    wins = Windows.from_arrays(
        start=0,
        stop=10,
        ref=[0, 5],
    )

    matches = wins.match_events([2, 7, 12, 20])

    assert isinstance(matches, WindowMatches)
    assert set(zip(matches.event_pos, matches.win_pos, strict=True)) == {
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 1),
    }


def test_match_events_respects_half_open_boundaries_and_empty_windows():
    wins = Windows.from_arrays(
        start=[0, 10, 5],
        stop=[10, 20, 5],
    )

    matches = wins.match_events([0, 10, 20, 5])

    assert set(zip(matches.event_pos, matches.win_pos, strict=True)) == {
        (0, 0),
        (1, 1),
        (3, 0),
    }


def test_classify_events_preserves_event_identity_and_adds_requested_projection():
    meta = pd.DataFrame(
        {'state': ['a', 'b']},
        index=pd.Index([10, 20], name='win_id'),
    )
    wins = Windows.from_arrays(0, 10, [0, 5], meta=meta)
    events = pd.Series(
        [2, 7, 12],
        index=pd.Index([100, 101, 102], name='event_id'),
    )

    result = wins.classify_events(
        events,
        relative_to='ref',
        cols='state',
    )

    expected = pd.DataFrame(
        {
            'item_id': [10, 10, 20, 20],
            'delay': [2.0, 7.0, 2.0, 7.0],
            'state': ['a', 'a', 'b', 'b'],
        },
        index=pd.Index([100, 101, 101, 102], name='event_id'),
    )
    pd.testing.assert_frame_equal(result, expected)


def test_classify_events_only_computes_requested_columns():
    wins = Windows.from_arrays(0, 10, [0, 20])

    result = wins.classify_events([5, 25])

    assert list(result.columns) == ['item_id']
    assert list(result['item_id']) == [0, 1]


def test_generate_preserves_input_identity_and_fills_unmatched_times():
    meta = pd.DataFrame({'state': ['a', 'b']}, index=pd.Index([0, 1], name='win'))
    wins = Windows.from_arrays(
        start=[0, 10],
        stop=[10, 20],
        meta=meta,
    )
    times = pd.Series(
        [0, 9, 10, 20, 30],
        index=pd.Index([50, 51, 52, 53, 54], name='event_id'),
    )

    result = wins.generate(times, by='state', fill_value='none')

    pd.testing.assert_series_equal(
        result,
        pd.Series(
            ['a', 'a', 'b', 'none', 'none'],
            index=times.index,
            name='state',
        ),
    )


def test_generate_requires_exclusive_windows():
    wins = Windows.from_arrays(
        [0, 5],
        [10, 15],
        meta=pd.DataFrame({'state': ['a', 'b']}, index=pd.Index([0, 1], name='win')),
    )

    with pytest.raises(ValueError):
        wins.generate([7], by='state')


def test_is_sandwiched_is_index_aligned_and_supports_filters():
    meta = pd.DataFrame(
        {'state': ['a', 'b', 'a']},
        index=pd.Index([10, 20, 30], name='win_id'),
    )
    wins = Windows.from_arrays(
        start=[0, 10, 12],
        stop=[10, 12, 22],
        meta=meta,
    )

    pd.testing.assert_series_equal(
        wins.is_sandwiched('state'),
        pd.Series([False, True, False], index=meta.index, name='is_sandwiched'),
    )
    assert not wins.is_sandwiched('state', max_length=1).any()
    assert wins.is_sandwiched('state', only='b').loc[20]
    assert not wins.is_sandwiched('state', only='a').any()


def test_merge_sandwiched_relabels_merges_and_preserves_provenance():
    meta = pd.DataFrame(
        {'state': ['a', 'b', 'a']},
        index=pd.Index([10, 20, 30], name='win_id'),
    )
    wins = Windows.from_arrays(
        start=[0, 10, 12],
        stop=[10, 12, 22],
        meta=meta,
    )

    result = wins.merge_sandwiched('state')

    assert len(result) == 1
    assert result.meta.iloc[0]['state'] == 'a'
    assert result.meta.iloc[0]['source_win_ids'] == (10, 20, 30)
    assert result.time_at('start').iloc[0] == 0
    assert result.time_at('stop').iloc[0] == 22


# -----------------------------------------------------------------------------
# representation


def test_to_frame_combines_geometry_and_metadata_without_aliasing_metadata():
    meta = pd.DataFrame(
        {'state': ['a', 'b']},
        index=pd.Index([10, 20], name='win_id'),
    )
    wins = Windows.from_arrays(
        [0, 1],
        [10, 11],
        [100, 200],
        meta=meta,
    )

    frame = wins.to_frame()

    assert list(frame.columns) == ['start', 'stop', 'ref', 'state']
    assert frame.index.equals(meta.index)
    frame.loc[10, 'state'] = 'changed'
    assert wins.meta.loc[10, 'state'] == 'a'
