import numpy as np
import pandas as pd
import pytest

from nocte._coll import events, traces, trains, windows
from nocte._core import matching

# ----------------------------------------------------------------------
# fixtures


def _cuts() -> windows.Windows:
    meta = pd.DataFrame(
        {
            'animal': ['b', 'a'],
            'condition': ['early', 'late'],
        },
        index=pd.Index(
            [100, 200],
            name='win',
        ),
    )

    return windows.Windows.from_arrays(
        start=[-10.0, -10.0],
        stop=[20.0, 20.0],
        ref=[30.0, 70.0],
        meta=meta,
    )


def _no_match_cut() -> windows.Windows:
    meta = pd.DataFrame(
        {'animal': ['c']},
        index=pd.Index(
            [300],
            name='win',
        ),
    )

    return windows.Windows.from_arrays(
        start=-10.0,
        stop=20.0,
        ref=50.0,
        meta=meta,
    )


def _traces() -> traces.Traces:
    time = np.arange(
        0.0,
        110.0,
        10.0,
    )

    values = np.vstack(
        [
            time,
            1000.0 + time,
        ]
    )

    meta = pd.DataFrame(
        {
            'animal': ['a', 'b'],
            'channel': ['left', 'right'],
        },
        index=pd.Index(
            [10, 20],
            name='trace',
        ),
    )

    return traces.Traces.from_array(
        values,
        100.0,
        start=0.0,
        meta=meta,
    )


def _segments() -> windows.Windows:
    meta = pd.DataFrame(
        {
            'animal': ['a', 'b'],
            'kind': ['first', 'second'],
        },
        index=pd.Index(
            [10, 20],
            name='segment',
        ),
    )

    return windows.Windows.from_arrays(
        start=[0.0, 0.0],
        stop=[100.0, 100.0],
        ref=[0.0, 0.0],
        meta=meta,
    )


def _events() -> events.Events:
    meta = pd.DataFrame(
        {
            'animal': ['b', 'a'],
            'kind': ['first', 'second'],
        },
        index=pd.Index(
            [10, 20],
            name='event',
        ),
    )

    return events.Events.from_times(
        [30.0, 70.0],
        meta=meta,
    )


def _trains() -> trains.Trains:
    meta = pd.DataFrame(
        {
            'animal': ['b', 'a'],
            'kind': ['first', 'second'],
        },
        index=pd.Index(
            [10, 20],
            name='train',
        ),
    )

    return trains.Trains.from_times(
        {
            10: [25.0, 30.0, 35.0, 65.0, 70.0, 75.0],
            20: [26.0, 31.0, 36.0, 66.0, 71.0, 76.0],
        },
        support=windows.Win(0.0, 100.0),
        meta=meta,
    )


# ----------------------------------------------------------------------
# helpers


def _values_by_id(
    obj: traces.Traces,
    ids: list[int],
) -> np.ndarray:
    positions = obj.index.get_indexer(pd.Index(ids))

    assert (positions >= 0).all()

    return obj.values[positions]


def _trace_for_source(
    obj: traces.Traces,
    column: str,
    source_id: int,
) -> np.ndarray:
    positions = np.flatnonzero(obj.meta[column].to_numpy() == source_id)

    assert len(positions) == 1

    return obj.values[int(positions[0])]


def _assert_provenance(
    obj,
    left: str,
    right: str,
    expected: list[tuple[int, int]],
) -> None:
    actual = list(
        obj.meta.loc[:, [left, right]].itertuples(
            index=False,
            name=None,
        )
    )

    assert sorted(actual) == sorted(expected)


def _support_bounds(
    support: windows.Win,
) -> tuple[float, float]:
    return (
        support.time_at('start'),
        support.time_at('stop'),
    )


# ----------------------------------------------------------------------
# specialized groupings


@pytest.mark.parametrize(
    ('factory', 'grouping_type'),
    [
        (_traces, traces.TracesGrouping),
        (_segments, windows.WindowsGrouping),
        (_events, events.EventsGrouping),
        (_trains, trains.TrainsGrouping),
    ],
    ids=[
        'traces',
        'windows',
        'events',
        'trains',
    ],
)
def test_groupby_returns_specialized_temporal_grouping(
    factory,
    grouping_type,
):
    grouped = factory().groupby('animal')

    assert isinstance(grouped, grouping_type)


# ----------------------------------------------------------------------
# Traces.extract_win


def test_traces_extract_win_interpolates_without_cropping_source():
    obj = _traces().sel_index([10])

    # Absolute interval [5, 25), aligned to ref=15.
    #
    # At 100 Hz the desired samples are at absolute [5, 15].
    # The value at 5 requires interpolation using the native sample
    # at t=0, outside the requested Window.
    win = windows.Win(
        -10.0,
        10.0,
        ref=15.0,
    )

    result = obj.extract_win(win)

    assert result.hz == 100.0
    assert result.start == -10.0
    assert result.n_samples == 2

    np.testing.assert_allclose(
        result.time,
        [-10.0, 0.0],
    )
    np.testing.assert_allclose(
        result.values,
        [[5.0, 15.0]],
    )
    pd.testing.assert_frame_equal(
        result.meta,
        obj.meta,
    )


def test_traces_extract_win_align_none_preserves_absolute_time():
    obj = _traces().sel_index([10])

    win = windows.Win(
        -10.0,
        10.0,
        ref=15.0,
    )

    result = obj.extract_win(
        win,
        hz=200.0,
        align=None,
    )

    assert result.hz == 200.0
    assert result.start == 5.0

    np.testing.assert_allclose(
        result.time,
        [5.0, 10.0, 15.0, 20.0],
    )
    np.testing.assert_allclose(
        result.values,
        [[5.0, 10.0, 15.0, 20.0]],
    )


def test_traces_extract_win_drop_controls_empty_traces():
    time = np.arange(
        0.0,
        110.0,
        10.0,
    )

    values = np.vstack(
        [
            time,
            1000.0 + time,
        ]
    )
    values[1, time < 60.0] = np.nan

    obj = traces.Traces.from_array(
        values,
        100.0,
        meta=pd.DataFrame(
            index=pd.Index(
                [10, 20],
                name='trace',
            )
        ),
    )

    win = windows.Win(
        -10.0,
        20.0,
        ref=30.0,
    )

    dropped = obj.extract_win(win)
    kept = obj.extract_win(
        win,
        drop=False,
    )

    assert dropped.index.tolist() == [10]

    assert kept.index.tolist() == [10, 20]
    assert kept.is_empty().tolist() == [False, True]


# ----------------------------------------------------------------------
# Windows.extract_win


def test_windows_extract_win_aligns_and_drops_disjoint_windows():
    obj = windows.Windows.from_arrays(
        start=[20.0, 80.0],
        stop=[40.0, 90.0],
        ref=[0.0, 0.0],
        meta=pd.DataFrame(
            index=pd.Index(
                [10, 20],
                name='segment',
            )
        ),
    )

    win = windows.Win(
        -10.0,
        20.0,
        ref=30.0,
    )

    result = obj.extract_win(win)

    assert result.index.tolist() == [10]

    np.testing.assert_allclose(
        result.time_at('start'),
        [-10.0],
    )
    np.testing.assert_allclose(
        result.time_at('stop'),
        [10.0],
    )


def test_windows_extract_win_align_none_and_drop_false_preserve_empty_items():
    obj = windows.Windows.from_arrays(
        start=[20.0, 80.0],
        stop=[40.0, 90.0],
        ref=[0.0, 0.0],
        meta=pd.DataFrame(
            index=pd.Index(
                [10, 20],
                name='segment',
            )
        ),
    )

    win = windows.Win(
        -10.0,
        20.0,
        ref=30.0,
    )

    result = obj.extract_win(
        win,
        align=None,
        drop=False,
    )

    assert result.index.tolist() == [10, 20]

    np.testing.assert_allclose(
        result.time_at('start'),
        [20.0, 50.0],
    )
    np.testing.assert_allclose(
        result.time_at('stop'),
        [40.0, 50.0],
    )

    assert result.is_empty().tolist() == [False, True]


# ----------------------------------------------------------------------
# Events.extract_win


def test_events_extract_win_is_half_open_and_aligns_to_reference():
    obj = events.Events.from_times(
        pd.Series(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            index=pd.Index(
                [10, 20, 30, 40, 50],
                name='event',
            ),
        )
    )

    win = windows.Win(
        -1.0,
        2.0,
        ref=2.0,
    )

    result = obj.extract_win(win)

    assert result.index.tolist() == [20, 30, 40]
    assert result.time.tolist() == [-1.0, 0.0, 1.0]


def test_events_extract_win_align_none_preserves_absolute_time():
    obj = events.Events.from_times(
        pd.Series(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            index=pd.Index(
                [10, 20, 30, 40, 50],
                name='event',
            ),
        )
    )

    win = windows.Win(
        -1.0,
        2.0,
        ref=2.0,
    )

    result = obj.extract_win(
        win,
        align=None,
    )

    assert result.index.tolist() == [20, 30, 40]
    assert result.time.tolist() == [1.0, 2.0, 3.0]


def test_events_extract_win_can_return_empty_collection():
    obj = events.Events.from_times(
        [1.0, 2.0],
    )

    result = obj.extract_win(
        windows.Win(10.0, 20.0),
    )

    assert len(result) == 0
    assert result.index.name == obj.index.name


# ----------------------------------------------------------------------
# Trains.extract_win


def test_trains_extract_win_aligns_support_and_drops_silent_trains():
    obj = trains.Trains.from_times(
        {
            10: [25.0, 30.0, 35.0],
            20: [5.0],
        },
        support=windows.Win(0.0, 100.0),
    )

    win = windows.Win(
        -10.0,
        20.0,
        ref=30.0,
    )

    result = obj.extract_win(win)

    assert result.index.tolist() == [10]
    assert _support_bounds(result.support) == pytest.approx((-10.0, 20.0))

    np.testing.assert_allclose(
        result.get(10),
        [-5.0, 0.0, 5.0],
    )


def test_trains_extract_win_align_none_and_drop_false_preserve_silent_trains():
    obj = trains.Trains.from_times(
        {
            10: [25.0, 30.0, 35.0],
            20: [5.0],
        },
        support=windows.Win(0.0, 100.0),
    )

    win = windows.Win(
        -10.0,
        20.0,
        ref=30.0,
    )

    result = obj.extract_win(
        win,
        align=None,
        drop=False,
    )

    assert result.index.tolist() == [10, 20]
    assert _support_bounds(result.support) == pytest.approx((20.0, 50.0))

    np.testing.assert_allclose(
        result.get(10),
        [25.0, 30.0, 35.0],
    )
    assert result.get(20).size == 0


# ----------------------------------------------------------------------
# Traces.extract_matched


def test_traces_extract_matched_uses_common_grid_across_windows():
    obj = _traces()

    wins = windows.Windows.from_arrays(
        start=[-20.0, -10.0],
        stop=[10.0, 30.0],
        ref=[30.0, 70.0],
        meta=pd.DataFrame(
            index=pd.Index(
                [100, 200],
                name='win',
            )
        ),
    )

    matches = matching.Matches.from_product(
        obj,
        wins,
    ).sel_index([0, 3])

    result = obj.extract_matched(
        wins,
        matches,
    )

    assert result.hz == 100.0
    assert result.start == -20.0
    assert result.stop == 30.0

    np.testing.assert_allclose(
        result.time,
        [-20.0, -10.0, 0.0, 10.0, 20.0],
    )

    values = _values_by_id(
        result,
        [0, 3],
    )

    zero = int(
        np.flatnonzero(
            np.isclose(
                result.time,
                0.0,
            )
        )[0]
    )

    np.testing.assert_allclose(
        values[:, zero],
        [
            30.0,
            1070.0,
        ],
    )

    _assert_provenance(
        result,
        'trace',
        'win',
        [
            (10, 100),
            (20, 200),
        ],
    )


# ----------------------------------------------------------------------
# Windows.extract_matched


def test_windows_extract_matched_uses_explicit_relation_and_alignment():
    obj = _segments()
    wins = _cuts()

    matches = matching.Matches.from_product(
        obj,
        wins,
    ).sel_index([0, 3])

    result = obj.extract_matched(
        wins,
        matches,
        align='start',
    )

    assert len(result) == 2

    np.testing.assert_allclose(
        result.time_at('start'),
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        result.time_at('stop'),
        [30.0, 30.0],
    )

    _assert_provenance(
        result,
        'segment',
        'win',
        [
            (10, 100),
            (20, 200),
        ],
    )


# ----------------------------------------------------------------------
# Events.extract_matched


def test_events_extract_matched_uses_explicit_relation_and_alignment():
    obj = _events()
    wins = _cuts()

    matches = matching.Matches.from_product(
        obj,
        wins,
    ).sel_index([0, 3])

    result = obj.extract_matched(
        wins,
        matches,
        align='start',
    )

    assert len(result) == 2
    assert result.time.tolist() == [10.0, 10.0]

    _assert_provenance(
        result,
        'event',
        'win',
        [
            (10, 100),
            (20, 200),
        ],
    )


# ----------------------------------------------------------------------
# Trains.extract_matched


def test_trains_extract_matched_uses_explicit_relation_and_alignment():
    obj = _trains()
    wins = _cuts()

    matches = matching.Matches.from_product(
        obj,
        wins,
    ).sel_index([0, 3])

    result = obj.extract_matched(
        wins,
        matches,
        align='start',
    )

    assert len(result) == 2
    assert result.counts().tolist() == [3, 3]

    assert _support_bounds(result.support) == pytest.approx((0.0, 30.0))

    _assert_provenance(
        result,
        'train',
        'win',
        [
            (10, 100),
            (20, 200),
        ],
    )


# ----------------------------------------------------------------------
# extract_all


def test_traces_extract_all_extracts_cartesian_product():
    obj = _traces()
    wins = _cuts()

    result = obj.extract_all(wins)

    assert len(result) == 4

    np.testing.assert_allclose(
        result.time,
        [-10.0, 0.0, 10.0],
    )

    values = _values_by_id(
        result,
        [0, 1, 2, 3],
    )

    np.testing.assert_allclose(
        values,
        [
            [20.0, 30.0, 40.0],
            [60.0, 70.0, 80.0],
            [1020.0, 1030.0, 1040.0],
            [1060.0, 1070.0, 1080.0],
        ],
    )

    _assert_provenance(
        result,
        'trace',
        'win',
        [
            (10, 100),
            (10, 200),
            (20, 100),
            (20, 200),
        ],
    )


def test_windows_extract_all_extracts_every_overlapping_pair():
    obj = _segments()
    wins = _cuts()

    result = obj.extract_all(wins)

    assert len(result) == 4

    np.testing.assert_allclose(
        result.time_at('start'),
        [-10.0] * 4,
    )
    np.testing.assert_allclose(
        result.time_at('stop'),
        [20.0] * 4,
    )

    _assert_provenance(
        result,
        'segment',
        'win',
        [
            (10, 100),
            (10, 200),
            (20, 100),
            (20, 200),
        ],
    )


def test_events_extract_all_returns_only_actual_containment_pairs():
    obj = _events()
    wins = _cuts()

    result = obj.extract_all(wins)

    # Events are atomic. There are four possible Cartesian pairs,
    # but only two actual event-in-window observations.
    assert len(result) == 2
    assert result.time.tolist() == [0.0, 0.0]

    _assert_provenance(
        result,
        'event',
        'win',
        [
            (10, 100),
            (20, 200),
        ],
    )


def test_trains_extract_all_extracts_cartesian_product_on_common_support():
    obj = _trains()
    wins = _cuts()

    result = obj.extract_all(wins)

    assert len(result) == 4
    assert result.counts().tolist() == [3, 3, 3, 3]

    assert _support_bounds(result.support) == pytest.approx((-10.0, 20.0))

    _assert_provenance(
        result,
        'train',
        'win',
        [
            (10, 100),
            (10, 200),
            (20, 100),
            (20, 200),
        ],
    )


# ----------------------------------------------------------------------
# extract_all drop policy


def test_traces_extract_all_drop_controls_empty_matches():
    time = np.arange(
        0.0,
        110.0,
        10.0,
    )
    values = time.reshape(1, -1)
    values = values.astype(float)
    values[:, time >= 60.0] = np.nan

    obj = traces.Traces.from_array(
        values,
        100.0,
        meta=pd.DataFrame(
            index=pd.Index(
                [10],
                name='trace',
            )
        ),
    )

    dropped = obj.extract_all(_cuts())
    kept = obj.extract_all(
        _cuts(),
        drop=False,
    )

    assert len(dropped) == 1

    assert len(kept) == 2
    assert kept.is_empty().sum() == 1


def test_windows_extract_all_drop_controls_empty_matches():
    obj = windows.Windows.from_arrays(
        start=20.0,
        stop=50.0,
        ref=0.0,
        meta=pd.DataFrame(
            index=pd.Index(
                [10],
                name='segment',
            )
        ),
    )

    dropped = obj.extract_all(_cuts())
    kept = obj.extract_all(
        _cuts(),
        drop=False,
    )

    assert len(dropped) == 1

    assert len(kept) == 2
    assert kept.is_empty().sum() == 1


def test_trains_extract_all_drop_controls_silent_matches():
    obj = trains.Trains.from_times(
        {
            10: [25.0, 30.0, 35.0],
        },
        support=windows.Win(0.0, 100.0),
    )

    dropped = obj.extract_all(_cuts())
    kept = obj.extract_all(
        _cuts(),
        drop=False,
    )

    assert len(dropped) == 1

    assert len(kept) == 2
    assert sorted(kept.counts().tolist()) == [0, 3]


# ----------------------------------------------------------------------
# extract_by


def test_traces_extract_by_uses_metadata_relation():
    obj = _traces()
    wins = _cuts()

    result = obj.extract_by(
        wins,
        by='animal',
    )

    assert len(result) == 2

    np.testing.assert_allclose(
        _trace_for_source(
            result,
            'trace',
            10,
        ),
        [60.0, 70.0, 80.0],
    )
    np.testing.assert_allclose(
        _trace_for_source(
            result,
            'trace',
            20,
        ),
        [1020.0, 1030.0, 1040.0],
    )

    _assert_provenance(
        result,
        'trace',
        'win',
        [
            (10, 200),
            (20, 100),
        ],
    )

    assert sorted(result.meta['animal'].tolist()) == ['a', 'b']


def test_windows_extract_by_uses_metadata_relation():
    obj = _segments()
    wins = _cuts()

    result = obj.extract_by(
        wins,
        by='animal',
    )

    assert len(result) == 2

    np.testing.assert_allclose(
        result.time_at('start'),
        [-10.0, -10.0],
    )
    np.testing.assert_allclose(
        result.time_at('stop'),
        [20.0, 20.0],
    )

    _assert_provenance(
        result,
        'segment',
        'win',
        [
            (10, 200),
            (20, 100),
        ],
    )


def test_events_extract_by_uses_metadata_relation():
    obj = _events()
    wins = _cuts()

    result = obj.extract_by(
        wins,
        by='animal',
    )

    assert len(result) == 2
    assert result.time.tolist() == [0.0, 0.0]

    _assert_provenance(
        result,
        'event',
        'win',
        [
            (10, 100),
            (20, 200),
        ],
    )


def test_trains_extract_by_uses_metadata_relation():
    obj = _trains()
    wins = _cuts()

    result = obj.extract_by(
        wins,
        by='animal',
    )

    assert len(result) == 2
    assert result.counts().tolist() == [3, 3]

    assert _support_bounds(result.support) == pytest.approx((-10.0, 20.0))

    _assert_provenance(
        result,
        'train',
        'win',
        [
            (10, 100),
            (20, 200),
        ],
    )


@pytest.mark.parametrize(
    'factory',
    [
        _traces,
        _segments,
        _events,
        _trains,
    ],
    ids=[
        'traces',
        'windows',
        'events',
        'trains',
    ],
)
def test_extract_by_no_matches_returns_empty_collection(factory):
    result = factory().extract_by(
        _no_match_cut(),
        by='animal',
    )

    assert len(result) == 0
    assert result.index.name == 'match'


# ----------------------------------------------------------------------
# same-name Windows relation


def test_windows_extract_all_disambiguates_same_name_provenance():
    obj = windows.Windows.from_arrays(
        start=0.0,
        stop=100.0,
        ref=0.0,
    )

    wins = windows.Windows.from_arrays(
        start=-10.0,
        stop=20.0,
        ref=30.0,
    )

    result = obj.extract_all(wins)

    assert len(result) == 1
    assert 'left_win' in result.meta.columns
    assert 'right_win' in result.meta.columns

    _assert_provenance(
        result,
        'left_win',
        'right_win',
        [
            (0, 0),
        ],
    )


# ----------------------------------------------------------------------
# Trains support compatibility


def test_trains_extract_matched_reports_mismatched_edge_clipped_supports():
    obj = _trains()

    wins = windows.Windows.from_arrays(
        start=[-10.0, -10.0],
        stop=[20.0, 20.0],
        ref=[5.0, 50.0],
        meta=pd.DataFrame(
            index=pd.Index(
                [100, 200],
                name='win',
            )
        ),
    )

    matches = matching.Matches.from_product(
        obj,
        wins,
    ).sel_index([0, 3])

    with pytest.raises(
        ValueError,
        match='all grouped Trains must share the same support',
    ) as exc_info:
        obj.extract_matched(
            wins,
            matches,
            drop=False,
        )

    message = str(exc_info.value)

    assert '1 x (-5, 20)' in message
    assert '1 x (-10, 20)' in message
