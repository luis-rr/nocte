import numpy as np
import pandas as pd

from nocte.core.matching import Matches
from nocte.core.traces import Traces
from nocte.core.windows import Win, Windows


def _traces() -> Traces:
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

    return Traces.from_array(
        values,
        100.0,
        start=0.0,
        meta=meta,
    )


def _windows() -> Windows:
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

    return Windows.from_arrays(
        start=[-10.0, -10.0],
        stop=[20.0, 20.0],
        ref=[30.0, 70.0],
        meta=meta,
    )


def _values_by_id(
    traces: Traces,
    ids: list[int],
) -> np.ndarray:
    positions = traces.index.get_indexer(pd.Index(ids))

    assert (positions >= 0).all()

    return traces.values[positions]


# ----------------------------------------------------------------------
# extract_win


def test_extract_win_interpolates_without_cropping_source():
    traces = _traces().sel_index([10])

    # Absolute interval [5, 25), aligned to ref=15.
    #
    # At 100 Hz the desired samples are at absolute times [5, 15].
    # The value at 5 requires interpolation using the native sample at
    # t=0, which lies outside the Window. This specifically guards against
    # implementing extraction as crop -> resample.
    win = Win(
        -10.0,
        10.0,
        ref=15.0,
    )

    result = traces.extract_win(
        win,
    )

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
        traces.meta,
    )


def test_extract_win_resamples_and_aligns_to_start():
    traces = _traces().sel_index([10])

    # Absolute Window [5, 25).
    win = Win(
        -10.0,
        10.0,
        ref=15.0,
    )

    result = traces.extract_win(
        win,
        hz=200.0,
        align='start',
    )

    assert result.hz == 200.0
    assert result.start == 0.0
    assert result.n_samples == 4

    np.testing.assert_allclose(
        result.time,
        [0.0, 5.0, 10.0, 15.0],
    )

    np.testing.assert_allclose(
        result.values,
        [[5.0, 10.0, 15.0, 20.0]],
    )


# ----------------------------------------------------------------------
# extract_matched


def test_extract_matched_uses_common_grid_across_windows():
    traces = _traces()

    wins = Windows.from_arrays(
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

    # Product is:
    #
    # match 0: trace 10 -> win 100
    # match 1: trace 10 -> win 200
    # match 2: trace 20 -> win 100
    # match 3: trace 20 -> win 200
    #
    # Keep one match for each Window.
    matches = Matches.from_product(
        traces,
        wins,
    ).sel_index([0, 3])

    result = traces.extract_matched(
        wins,
        matches,
    )

    # Relative Window extents are:
    #
    # win 100: [-20, 10)
    # win 200: [-10, 30)
    #
    # therefore the common zero-phased grid is [-20, 30).
    assert result.hz == 100.0
    assert result.start == -20.0
    assert result.stop == 30.0

    np.testing.assert_allclose(
        result.time,
        [-20.0, -10.0, 0.0, 10.0, 20.0],
    )

    # Regardless of the different Window geometry, t=0 must evaluate at
    # the requested alignment point.
    zero = int(
        np.flatnonzero(
            np.isclose(
                result.time,
                0.0,
            )
        )[0]
    )

    values = _values_by_id(
        result,
        [0, 3],
    )

    np.testing.assert_allclose(
        values[:, zero],
        [
            30.0,
            1070.0,
        ],
    )

    expected_provenance = pd.DataFrame(
        {
            'trace': [10, 20],
            'win': [100, 200],
        },
        index=pd.Index(
            [0, 3],
            name='match',
        ),
    )

    pd.testing.assert_frame_equal(
        result.meta.loc[
            [0, 3],
            ['trace', 'win'],
        ],
        expected_provenance,
    )


# ----------------------------------------------------------------------
# extract_all


def test_extract_all_extracts_cartesian_product():
    traces = _traces()
    wins = _windows()

    result = traces.extract_all(
        wins,
    )

    assert len(result) == 4

    np.testing.assert_allclose(
        result.time,
        [-10.0, 0.0, 10.0],
    )

    # Canonicalize by match ID so the test does not depend on the order in
    # which groups are flattened during concat.
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

    expected_provenance = pd.DataFrame(
        {
            'trace': [10, 10, 20, 20],
            'win': [100, 200, 100, 200],
        },
        index=pd.Index(
            [0, 1, 2, 3],
            name='match',
        ),
    )

    pd.testing.assert_frame_equal(
        result.meta.loc[
            [0, 1, 2, 3],
            ['trace', 'win'],
        ],
        expected_provenance,
    )

    # Source and Window metadata survive the relation/grouping/concat
    # round-trip.
    pd.testing.assert_series_equal(
        result.meta.loc[
            [0, 1, 2, 3],
            'channel',
        ],
        pd.Series(
            ['left', 'left', 'right', 'right'],
            index=pd.Index(
                [0, 1, 2, 3],
                name='match',
            ),
            name='channel',
        ),
    )

    pd.testing.assert_series_equal(
        result.meta.loc[
            [0, 1, 2, 3],
            'condition',
        ],
        pd.Series(
            ['early', 'late', 'early', 'late'],
            index=pd.Index(
                [0, 1, 2, 3],
                name='match',
            ),
            name='condition',
        ),
    )


# ----------------------------------------------------------------------
# extract_by


def test_extract_by_uses_many_to_many_metadata_matching():
    traces = _traces()
    wins = _windows()

    result = traces.extract_by(
        wins,
        by='animal',
    )

    # animal='a': trace 10 -> win 200
    # animal='b': trace 20 -> win 100
    assert len(result) == 2

    np.testing.assert_allclose(
        result.time,
        [-10.0, 0.0, 10.0],
    )

    values = _values_by_id(
        result,
        [0, 1],
    )

    np.testing.assert_allclose(
        values,
        [
            [60.0, 70.0, 80.0],
            [1020.0, 1030.0, 1040.0],
        ],
    )

    expected_provenance = pd.DataFrame(
        {
            'trace': [10, 20],
            'win': [200, 100],
            'animal': ['a', 'b'],
        },
        index=pd.Index(
            [0, 1],
            name='match',
        ),
    )

    pd.testing.assert_frame_equal(
        result.meta.loc[
            [0, 1],
            ['trace', 'win', 'animal'],
        ],
        expected_provenance,
    )


def test_extract_by_handles_no_matches():
    traces = _traces()

    wins = Windows.from_arrays(
        start=-10.0,
        stop=20.0,
        ref=50.0,
        meta=pd.DataFrame(
            {
                'animal': ['c'],
            },
            index=pd.Index(
                [100],
                name='win',
            ),
        ),
    )

    result = traces.extract_by(
        wins,
        by='animal',
    )

    assert result.shape == (0, 3)
    assert result.hz == 100.0
    assert result.start == -10.0
    assert result.stop == 20.0
    assert result.index.name == 'match'

    np.testing.assert_allclose(
        result.time,
        [-10.0, 0.0, 10.0],
    )
