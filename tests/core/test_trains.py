import typing

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from nocte.core import events, trains, windows


def test_from_times_builds_one_collection_item_per_train():
    obj = trains.Trains.from_times(
        [[1.0, 2.0, 3.0], [], [5.0]],
        support=windows.Win(0.0, 10.0),
    )

    assert len(obj) == 3
    assert obj.index.equals(pd.RangeIndex(3, name='train'))
    assert obj.counts().tolist() == [3, 0, 1]


def test_mapping_uses_integer_keys_as_train_identity():
    obj = trains.Trains.from_times(
        {20: [2.0], 10: [1.0, 3.0]},
        support=windows.Win(0.0, 10.0),
    )

    assert obj.index.tolist() == [20, 10]
    np.testing.assert_array_equal(obj.get(20), [2.0])
    np.testing.assert_array_equal(obj.get(10), [1.0, 3.0])


def test_mapping_with_explicit_meta_is_reordered_to_meta_identity():
    meta = pd.DataFrame(
        {'kind': ['a', 'b']},
        index=pd.Index([10, 20], name='train_id'),
    )

    obj = trains.Trains.from_times(
        {20: [2.0], 10: [1.0]},
        support=windows.Win(0.0, 10.0),
        meta=meta,
    )

    assert obj.index.tolist() == [10, 20]
    np.testing.assert_array_equal(obj.get(10), [1.0])
    np.testing.assert_array_equal(obj.get(20), [2.0])


def test_mapping_rejects_noninteger_keys_and_key_meta_mismatch():
    invalid_keys: typing.Any = {
        'unit-a': [1.0],
    }

    with pytest.raises(ValueError, match='integers'):
        trains.Trains.from_times(
            invalid_keys,
            support=windows.Win(0.0, 10.0),
        )

    with pytest.raises(ValueError, match='match meta index exactly'):
        trains.Trains.from_times(
            {10: [1.0]},
            support=windows.Win(0.0, 10.0),
            meta=pd.DataFrame(index=[20]),
        )


def test_payload_requires_sorted_finite_timestamps():
    support = windows.Win(0.0, 10.0)

    with pytest.raises(ValueError, match='monotonically non-decreasing'):
        trains.Trains.from_times([[2.0, 1.0]], support=support)

    with pytest.raises(ValueError, match='finite'):
        trains.Trains.from_times([[1.0, np.nan]], support=support)


def test_duplicate_timestamps_and_empty_trains_are_valid():
    obj = trains.Trains.from_times(
        [[1.0, 1.0, 2.0], []],
        support=windows.Win(0.0, 10.0),
    )

    assert obj.counts().tolist() == [3, 0]
    np.testing.assert_array_equal(obj.get(0), [1.0, 1.0, 2.0])
    assert obj.get(1).shape == (0,)


def test_support_is_half_open_and_rejects_timestamps_at_stop():
    support = windows.Win(0.0, 10.0)

    trains.Trains.from_times([[0.0, 9.999]], support=support)

    with pytest.raises(ValueError, match='outside support'):
        trains.Trains.from_times([[10.0]], support=support)

    with pytest.raises(ValueError, match='outside support'):
        trains.Trains.from_times([[-0.001]], support=support)


def test_support_validation_respects_window_reference():
    support = windows.Win(-10.0, 10.0, ref=100.0)

    obj = trains.Trains.from_times([[90.0, 100.0, 109.0]], support=support)

    assert obj.support == support


def test_get_returns_read_only_numpy_array():
    obj = trains.Trains.from_times(
        [[1.0, 2.0]],
        support=windows.Win(0.0, 10.0),
    )

    item = obj.get(0)

    assert isinstance(item, np.ndarray)
    assert not item.flags.writeable
    with pytest.raises(ValueError):
        item[0] = 99.0


def test_selection_preserves_train_identity_and_shares_immutable_train_arrays():
    obj = trains.Trains.from_times(
        {10: [1.0], 20: [2.0, 3.0], 30: []},
        support=windows.Win(0.0, 10.0),
    )

    selected = obj.sel_index([20, 10])

    assert selected.index.tolist() == [20, 10]
    assert selected._data.values[0] is obj._data.values[1]
    assert selected._data.values[1] is obj._data.values[0]


def test_copy_shares_payload_but_not_metadata():
    obj = trains.Trains.from_times(
        [[1.0], [2.0]],
        support=windows.Win(0.0, 10.0),
        meta=pd.DataFrame({'kind': ['a', 'b']}, index=pd.Index([10, 20], name='train')),
    )

    copied = obj.copy()
    copied.meta.loc[10, 'kind'] = 'changed'

    assert copied._data is obj._data
    assert copied.support == obj.support
    assert obj.meta.loc[10, 'kind'] == 'a'


def test_counts_and_rates_have_one_value_per_train():
    obj = trains.Trains.from_times(
        [[0.0, 100.0, 200.0, 300.0], [], [0.0, 1000.0]],
        support=windows.Win(0.0, 2000.0),
        meta=pd.DataFrame(index=pd.Index([10, 20, 30], name='train_id')),
    )

    pdt.assert_series_equal(
        obj.counts(),
        pd.Series([4, 0, 2], index=obj.index, name='count'),
    )
    pdt.assert_series_equal(
        obj.rates(),
        pd.Series([2.0, 0.0, 1.0], index=obj.index, name='rate'),
    )


def test_rates_are_undefined_for_zero_duration_support():
    obj = trains.Trains.from_times(
        [[]],
        support=windows.Win(5.0, 5.0),
    )

    with pytest.raises(ValueError, match='undefined'):
        obj.rates()


def test_intervals_are_computed_independently_within_each_train():
    obj = trains.Trains.from_times(
        {10: [1.0, 1.0, 4.0], 20: [2.0], 30: []},
        support=windows.Win(0.0, 10.0),
    )

    result = obj.intervals()

    np.testing.assert_array_equal(result[10], [0.0, 3.0])
    assert result[20].shape == (0,)
    assert result[30].shape == (0,)


def test_drop_silent_removes_only_empty_trains():
    obj = trains.Trains.from_times(
        {10: [1.0], 20: [], 30: [2.0]},
        support=windows.Win(0.0, 10.0),
    )

    result = obj.drop_silent()

    assert result.index.tolist() == [10, 30]
    assert result.counts().tolist() == [1, 1]


def test_shift_moves_timestamps_and_support_together():
    obj = trains.Trains.from_times(
        [[1.0, 4.0]],
        support=windows.Win(0.0, 10.0),
    )

    shifted = obj.shift(5.0)

    np.testing.assert_array_equal(shifted.get(0), [6.0, 9.0])
    assert shifted.support.time_at('start') == 5.0
    assert shifted.support.time_at('stop') == 15.0


def test_count_in_is_half_open_and_rate_in_normalizes_by_window_duration():
    obj = trains.Trains.from_times(
        {10: [0.0, 2.0, 5.0, 9.0], 20: [2.0, 4.0]},
        support=windows.Win(0.0, 10.0),
    )
    win = windows.Win(2.0, 5.0)

    counts = obj.count_in(win)
    rates = obj.rate_in(win)

    pdt.assert_series_equal(
        counts,
        pd.Series([1, 2], index=obj.index, name='count'),
    )
    expected_rates = pd.Series(
        [1000.0 / 3.0, 2000.0 / 3.0],
        index=obj.index,
        name='rate',
    )
    pdt.assert_series_equal(rates, expected_rates)


def test_count_and_rate_in_reject_unobserved_or_empty_windows():
    obj = trains.Trains.from_times(
        [[1.0]],
        support=windows.Win(0.0, 10.0),
    )

    with pytest.raises(ValueError, match='outside train support'):
        obj.count_in(windows.Win(-1.0, 2.0))

    with pytest.raises(ValueError, match='undefined'):
        obj.rate_in(windows.Win(2.0, 2.0))


def test_count_bins_returns_time_by_train_dataframe():
    obj = trains.Trains.from_times(
        {10: [0.0, 4.0, 5.0, 9.0], 20: [1.0, 6.0]},
        support=windows.Win(0.0, 10.0),
    )

    result = obj.count_bins([0.0, 5.0, 10.0])

    expected = pd.DataFrame(
        [[2, 1], [2, 1]],
        index=pd.IntervalIndex.from_breaks(
            [0.0, 5.0, 10.0],
            closed='left',
            name='time',
        ),
        columns=obj.index,
    )
    pdt.assert_frame_equal(result, expected)


def test_count_bins_rejects_range_outside_known_support():
    obj = trains.Trains.from_times(
        [[1.0]],
        support=windows.Win(0.0, 10.0),
    )

    with pytest.raises(ValueError, match='outside train support'):
        obj.count_bins([-1.0, 5.0])

    with pytest.raises(ValueError, match='outside train support'):
        obj.count_bins([5.0, 11.0])


def test_count_rolling_uses_support_by_default_and_preserves_train_columns():
    obj = trains.Trains.from_times(
        {10: [0.0, 1.0, 3.0, 7.0], 20: [4.0, 9.0]},
        support=windows.Win(0.0, 10.0),
    )

    result = obj.count_rolling(window=4.0, step=2.0)

    expected = pd.DataFrame(
        [
            [3, 0],
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        index=pd.Index([2.0, 4.0, 6.0, 8.0], name='time'),
        columns=obj.index,
    )
    pdt.assert_frame_equal(result, expected)


def test_rate_gaussian_matches_analytical_peak_and_silent_train_is_zero():
    obj = trains.Trains.from_times(
        {10: [1.0], 20: []},
        support=windows.Win(0.0, 2.0),
    )

    result = obj.rate_gaussian(
        sigma=1.0,
        step=1.0,
        width=1.0,
    )

    expected_hz = 1000.0 / np.sqrt(2.0 * np.pi)
    assert result.index.tolist() == [1.0]
    assert result.loc[1.0, 10] == pytest.approx(expected_hz)
    assert result.loc[1.0, 20] == 0.0


def test_to_frame_reports_train_level_summary_without_expanding_events():
    obj = trains.Trains.from_times(
        {10: [1.0, 2.0], 20: []},
        support=windows.Win(0.0, 1000.0),
        meta=pd.DataFrame({'kind': ['a', 'b']}, index=pd.Index([10, 20], name='train')),
    )

    frame = obj.to_frame()

    assert frame.index.tolist() == [10, 20]
    assert frame.columns.tolist() == ['count', 'rate', 'kind']
    assert frame['count'].tolist() == [2, 0]
    assert frame['rate'].tolist() == [2.0, 0.0]


def test_payload_flatten_uses_offsets_and_preserves_empty_trains():
    data = trains._TrainsData([[1.0, 2.0], [], [5.0]])

    flat, offsets = data.flatten()

    np.testing.assert_array_equal(flat, [1.0, 2.0, 5.0])
    np.testing.assert_array_equal(offsets, [0, 2, 2, 3])


def test_hdf_roundtrip_preserves_support_identity_metadata_and_empty_trains(tmp_path):
    obj = trains.Trains.from_times(
        {10: [91.0, 100.0], 20: [], 30: [109.0]},
        support=windows.Win(-10.0, 10.0, ref=100.0),
        meta=pd.DataFrame(
            {'kind': ['a', 'b', 'c']},
            index=pd.Index([10, 20, 30], name='train_id'),
        ),
    )
    path = tmp_path / 'trains.h5'

    obj.to_hdf(path, key='test_trains')
    loaded = trains.Trains.from_hdf(path, key='test_trains')

    pdt.assert_frame_equal(loaded.meta, obj.meta)
    assert loaded.support == obj.support
    assert loaded.index.equals(obj.index)
    for train_id in obj.index:
        np.testing.assert_array_equal(loaded.get(int(train_id)), obj.get(int(train_id)))
        assert not loaded.get(int(train_id)).flags.writeable


def test_singleton_train_and_events_share_count_bins_numerics():
    times = np.array([0.0, 1.0, 1.0, 4.0, 9.0])
    event_obj = events.Events.from_times(times)
    train_obj = trains.Trains.from_times(
        [times],
        support=windows.Win(0.0, 10.0),
    )
    bins = [0.0, 2.0, 5.0, 10.0]

    event_result = event_obj.count_bins(bins)
    train_result = train_obj.count_bins(bins).iloc[:, 0]
    train_result.name = 'count'

    pdt.assert_series_equal(event_result, train_result)


def test_singleton_train_and_events_share_rolling_count_numerics():
    times = np.array([0.0, 1.0, 3.0, 7.0, 9.0])
    within = windows.Win(0.0, 10.0)
    event_obj = events.Events.from_times(times)
    train_obj = trains.Trains.from_times([times], support=within)

    event_result = event_obj.count_rolling(4.0, step=2.0, within=within)
    train_result = train_obj.count_rolling(4.0, step=2.0).iloc[:, 0]
    train_result.name = 'count'

    pdt.assert_series_equal(event_result, train_result)


def test_singleton_train_and_events_share_gaussian_rate_numerics():
    times = np.array([1.0, 3.0, 7.0])
    within = windows.Win(0.0, 10.0)
    event_obj = events.Events.from_times(times)
    train_obj = trains.Trains.from_times([times], support=within)

    event_result = event_obj.rate_gaussian(
        sigma=1.0,
        step=2.0,
        within=within,
        width=2.0,
    )
    train_result = train_obj.rate_gaussian(
        sigma=1.0,
        step=2.0,
        width=2.0,
    ).iloc[:, 0]
    train_result.name = 'rate'

    pdt.assert_series_equal(event_result, train_result)
