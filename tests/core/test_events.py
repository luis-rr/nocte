import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from nocte._coll import events, windows


def test_from_times_builds_default_identity_and_float_payload():
    obj = events.Events.from_times([2, 1, 1])

    assert obj.index.equals(pd.RangeIndex(3, name='event'))
    pdt.assert_series_equal(
        obj.time,
        pd.Series([2.0, 1.0, 1.0], index=obj.index, name='time'),
    )
    assert len(obj) == 3


def test_from_times_series_preserves_event_identity():
    times = pd.Series(
        [10.0, 20.0],
        index=pd.Index([101, 205], name='event_id'),
    )

    obj = events.Events.from_times(times)

    assert obj.index.equals(times.index)
    pdt.assert_series_equal(obj.time, times.rename('time'))


def test_from_times_series_requires_matching_explicit_meta_index():
    times = pd.Series([1.0, 2.0], index=[10, 20])
    meta = pd.DataFrame({'kind': ['a', 'b']}, index=[20, 10])

    with pytest.raises(ValueError, match='meta index must match'):
        events.Events.from_times(times, meta=meta)


def test_duplicate_timestamps_are_distinct_events():
    meta = pd.DataFrame(index=pd.Index([10, 20, 30], name='event_id'))
    obj = events.Events.from_times([5.0, 5.0, 5.0], meta=meta)

    assert obj.index.tolist() == [10, 20, 30]
    assert obj.time.tolist() == [5.0, 5.0, 5.0]


def test_payload_is_immutable_and_public_time_is_a_copy():
    obj = events.Events.from_times([1.0, 2.0])

    assert not obj._time.flags.writeable

    public = obj.time
    public.iloc[0] = 99.0

    assert obj.time.tolist() == [1.0, 2.0]


def test_selection_preserves_identity_and_time_alignment():
    meta = pd.DataFrame(
        {'kind': ['a', 'b', 'c']},
        index=pd.Index([30, 10, 20], name='event_id'),
    )
    obj = events.Events.from_times([3.0, 1.0, 2.0], meta=meta)

    selected = obj.sel_index([20, 30])

    assert selected.index.tolist() == [20, 30]
    assert selected.time.tolist() == [2.0, 3.0]
    assert selected.meta['kind'].tolist() == ['c', 'a']


def test_get_returns_scalar_time_by_event_id():
    times = pd.Series([3.0, 7.0], index=pd.Index([100, 200], name='event'))
    obj = events.Events.from_times(times)

    assert obj.get(200) == 7.0


def test_copy_shares_immutable_payload_but_not_metadata():
    obj = events.Events.from_times(
        [1.0, 2.0],
        meta=pd.DataFrame({'kind': ['a', 'b']}, index=pd.Index([10, 20], name='event')),
    )

    copied = obj.copy()
    copied.meta.loc[10, 'kind'] = 'changed'

    assert copied._data is obj._data
    assert obj.meta.loc[10, 'kind'] == 'a'


def test_sort_time_is_stable_for_duplicate_timestamps():
    obj = events.Events.from_times(
        pd.Series([2.0, 1.0, 1.0], index=pd.Index([30, 10, 20], name='event'))
    )

    ascending = obj.sort_time()
    descending = obj.sort_time(ascending=False)

    assert ascending.index.tolist() == [10, 20, 30]
    assert descending.index.tolist() == [30, 10, 20]


def test_shift_scalar_preserves_identity_and_metadata():
    obj = events.Events.from_times(
        [1.0, 2.0],
        meta=pd.DataFrame({'kind': ['a', 'b']}, index=pd.Index([10, 20], name='event')),
    )

    shifted = obj.shift(5.0)

    assert shifted.index.equals(obj.index)
    assert shifted.time.tolist() == [6.0, 7.0]
    pdt.assert_frame_equal(shifted.meta, obj.meta)


def test_shift_series_aligns_by_event_identity_not_series_order():
    obj = events.Events.from_times(
        pd.Series([1.0, 2.0], index=pd.Index([10, 20], name='event'))
    )
    offsets = pd.Series([100.0, 1.0], index=[20, 10])

    shifted = obj.shift(offsets)

    assert shifted.time.tolist() == [2.0, 102.0]


def test_shift_rejects_wrong_length_or_nonfinite_offsets():
    obj = events.Events.from_times([1.0, 2.0])

    with pytest.raises(ValueError, match='one value per event'):
        obj.shift([1.0])

    with pytest.raises(ValueError, match='finite'):
        obj.shift([1.0, np.nan])


def test_round_supports_temporal_scales():
    obj = events.Events.from_times([1400.0, 1600.0])

    rounded = obj.round(scale='seconds')

    assert rounded.time.tolist() == [1000.0, 2000.0]


def test_contained_in_and_extract_win_use_half_open_window_semantics():
    obj = events.Events.from_times(
        pd.Series(
            [0.0, 1.0, 2.0, 3.0],
            index=pd.Index([10, 20, 30, 40], name='event'),
        )
    )
    win = windows.Win(1.0, 3.0)

    mask = obj.contained_in(win)
    cropped = obj.extract_win(win, align=None)

    pdt.assert_series_equal(
        mask,
        pd.Series(
            [False, True, True, False],
            index=obj.index,
            name='contained_in',
        ),
    )
    assert cropped.index.tolist() == [20, 30]
    assert cropped.time.tolist() == [1.0, 2.0]


def test_intervals_are_chronological_and_aligned_back_to_identity():
    obj = events.Events.from_times(
        pd.Series(
            [5.0, 1.0, 5.0, 3.0],
            index=pd.Index([10, 11, 12, 13], name='event_id'),
        )
    )

    result = obj.intervals()

    expected = pd.Series(
        [2.0, np.nan, 0.0, 2.0],
        index=obj.index,
        name='interval',
    )
    pdt.assert_series_equal(result, expected)


def test_intervals_empty_and_singleton_have_no_defined_interval():
    empty = events.Events.from_times([])
    singleton = events.Events.from_times([5.0])

    assert empty.intervals().empty
    assert np.isnan(singleton.intervals().iloc[0])


def test_count_bins_uses_uniform_half_open_semantics():
    obj = events.Events.from_times([0.0, 1.0, 2.0, 3.0])

    result = obj.count_bins([0.0, 1.0, 2.0, 3.0])

    expected = pd.Series(
        [1, 1, 1],
        index=pd.IntervalIndex.from_breaks(
            [0.0, 1.0, 2.0, 3.0],
            closed='left',
            name='time',
        ),
        name='count',
    )
    pdt.assert_series_equal(result, expected)


def test_count_bins_ignores_events_outside_requested_range():
    obj = events.Events.from_times([-1.0, 0.0, 0.5, 2.0])

    result = obj.count_bins([0.0, 1.0])

    assert result.tolist() == [2]


def test_count_rolling_has_known_centers_and_counts():
    obj = events.Events.from_times([0.0, 1.0, 3.0, 4.0, 7.0, 9.0, 10.0])

    result = obj.count_rolling(
        4.0,
        step=2.0,
        within=windows.Win(0.0, 10.0),
    )

    expected = pd.Series(
        [3, 2, 2, 2],
        index=pd.Index([2.0, 4.0, 6.0, 8.0], name='time'),
        name='count',
    )
    pdt.assert_series_equal(result, expected)


def test_count_rolling_returns_empty_series_when_window_does_not_fit():
    obj = events.Events.from_times([1.0])

    result = obj.count_rolling(
        20.0,
        step=1.0,
        within=windows.Win(0.0, 10.0),
    )

    assert result.empty
    assert result.index.name == 'time'


def test_rate_gaussian_matches_single_event_analytical_peak_in_hz():
    obj = events.Events.from_times([1.0])

    result = obj.rate_gaussian(
        sigma=1.0,
        step=1.0,
        within=windows.Win(0.0, 2.0),
        width=1.0,
    )

    expected_hz = 1000.0 / np.sqrt(2.0 * np.pi)
    assert result.index.tolist() == [1.0]
    assert result.iloc[0] == pytest.approx(expected_hz)


def test_rate_gaussian_empty_events_produce_zero_rate():
    obj = events.Events.from_times([])

    result = obj.rate_gaussian(
        sigma=1.0,
        step=1.0,
        within=windows.Win(0.0, 2.0),
        width=1.0,
    )

    assert result.tolist() == [0.0]


def test_to_frame_combines_structural_time_and_metadata():
    obj = events.Events.from_times(
        [1.0, 2.0],
        meta=pd.DataFrame({'kind': ['a', 'b']}, index=pd.Index([10, 20], name='event')),
    )

    frame = obj.to_frame()

    assert frame.columns.tolist() == ['time', 'kind']
    assert frame.index.tolist() == [10, 20]
    assert frame['time'].tolist() == [1.0, 2.0]


def test_to_frame_warns_if_metadata_reuses_structural_time_name():
    obj = events.Events.from_times(
        [1.0],
        meta=pd.DataFrame({'time': ['metadata']}, index=pd.Index([10], name='event')),
    )

    with pytest.warns(UserWarning, match='duplicate columns'):
        frame = obj.to_frame()

    assert frame.columns.tolist() == ['time', 'time']


def test_hdf_roundtrip_preserves_identity_metadata_and_time(tmp_path):
    obj = events.Events.from_times(
        [3.0, 1.0, 3.0],
        meta=pd.DataFrame(
            {'kind': ['a', 'b', 'c']},
            index=pd.Index([30, 10, 20], name='event_id'),
        ),
    )
    path = tmp_path / 'events.h5'

    obj.to_hdf(path, key='test_events')
    loaded = events.Events.from_hdf(path, key='test_events')

    pdt.assert_frame_equal(loaded.meta, obj.meta)
    pdt.assert_series_equal(loaded.time, obj.time)
    assert not loaded._time.flags.writeable
