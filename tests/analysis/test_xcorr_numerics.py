from __future__ import annotations

import numpy as np

from nocte.analysis import xcorr

RTOL = 1e-12
ATOL = 1e-12


def _sine(
    n_samples: int,
    period: int,
    *,
    phase: float = 0.0,
) -> np.ndarray:
    """Sample a unit-amplitude sine with an exact integer sample period."""
    samples = np.arange(
        n_samples,
        dtype=float,
    )
    return np.sin(2 * np.pi * samples / period + phase)


def _packed(
    left: np.ndarray,
    right: np.ndarray,
    offsets: np.ndarray,
    *,
    left_positions: np.ndarray | None = None,
    right_positions: np.ndarray | None = None,
    left_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    right_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> xcorr._PackedXCorrCore:
    """
    Build numerical-core inputs without involving Traces or Matches.

    Bounds are half-open sample indices.
    """
    left = np.asarray(
        left,
        dtype=np.float64,
    )
    right = np.asarray(
        right,
        dtype=np.float64,
    )

    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]

    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)

    if left_positions is None:
        left_positions = np.arange(
            min(
                len(left),
                len(right),
            ),
            dtype=np.intp,
        )

    if right_positions is None:
        right_positions = np.arange(
            min(
                len(left),
                len(right),
            ),
            dtype=np.intp,
        )

    left_positions = np.ascontiguousarray(
        left_positions,
        dtype=np.intp,
    )
    right_positions = np.ascontiguousarray(
        right_positions,
        dtype=np.intp,
    )

    if left_bounds is None:
        left_first = np.zeros(
            len(left),
            dtype=np.intp,
        )
        left_stop = np.full(
            len(left),
            left.shape[1],
            dtype=np.intp,
        )
    else:
        left_first, left_stop = left_bounds

    if right_bounds is None:
        right_first = np.zeros(
            len(right),
            dtype=np.intp,
        )
        right_stop = np.full(
            len(right),
            right.shape[1],
            dtype=np.intp,
        )
    else:
        right_first, right_stop = right_bounds

    return xcorr._PackedXCorrCore(
        left_values=left,
        right_values=right,
        left_positions=left_positions,
        right_positions=right_positions,
        left_first=np.ascontiguousarray(
            left_first,
            dtype=np.intp,
        ),
        left_stop=np.ascontiguousarray(
            left_stop,
            dtype=np.intp,
        ),
        right_first=np.ascontiguousarray(
            right_first,
            dtype=np.intp,
        ),
        right_stop=np.ascontiguousarray(
            right_stop,
            dtype=np.intp,
        ),
        offsets=np.ascontiguousarray(
            offsets,
            dtype=np.intp,
        ),
    )


def _metric(
    method: str,
    *,
    kernel: np.ndarray | None = None,
) -> xcorr._MetricCore:
    if method == 'pearson':
        code = xcorr._PEARSON
    elif method == 'dot':
        code = xcorr._DOT
    else:
        raise ValueError(method)

    if kernel is None:
        kernel = np.empty(
            0,
            dtype=np.float64,
        )

    return xcorr._MetricCore(
        method=code,
        kernel=np.ascontiguousarray(
            kernel,
            dtype=np.float64,
        ),
    )


def test_cross_corr_nb_sine_autocorrelation():
    """
    A sine shifted by half a period is its negative, while shifting by a full
    period reproduces it exactly.
    """
    period = 32
    signal = _sine(
        period * 20,
        period,
        phase=0.31,
    )

    offsets = np.array(
        [
            -period,
            -period // 2,
            0,
            period // 2,
            period,
        ],
        dtype=np.intp,
    )

    packed = _packed(
        signal,
        signal,
        offsets,
    )

    actual = xcorr._cross_corr_nb(
        packed,
        _metric('pearson'),
    )

    expected = np.array([[1.0, -1.0, 1.0, -1.0, 1.0]])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_cross_corr_nb_pearson_matches_analytic_sine_phase():
    """
    For equal-frequency sinusoids,

        corr(sin(wt + a), sin(w(t + d) + b))
            = cos((b - a) + wd)

    when evaluated over an integer number of periods.

    The left bounds are restricted to a fixed interior region while the right
    signal is padded, so every lag uses exactly the same number of samples.
    """
    period = 32
    omega = 2 * np.pi / period

    n_samples = period * 20
    padding = period // 2
    total_samples = n_samples + 2 * padding

    left_phases = np.array(
        [
            0.1,
            -0.4,
        ]
    )
    right_phases = np.array(
        [
            0.5,
            0.2,
            -0.7,
        ]
    )

    left = np.stack(
        [
            _sine(
                total_samples,
                period,
                phase=phase,
            )
            for phase in left_phases
        ]
    )
    right = np.stack(
        [
            _sine(
                total_samples,
                period,
                phase=phase,
            )
            for phase in right_phases
        ]
    )

    left_positions = np.array(
        [1, 0, 1],
        dtype=np.intp,
    )
    right_positions = np.array(
        [0, 2, 1],
        dtype=np.intp,
    )

    offsets = np.array(
        [-16, -8, 0, 8, 16],
        dtype=np.intp,
    )

    packed = _packed(
        left,
        right,
        offsets,
        left_positions=left_positions,
        right_positions=right_positions,
        left_bounds=(
            np.full(
                len(left),
                padding,
                dtype=np.intp,
            ),
            np.full(
                len(left),
                padding + n_samples,
                dtype=np.intp,
            ),
        ),
    )

    actual = xcorr._cross_corr_nb(
        packed,
        _metric('pearson'),
    )

    phase_difference = right_phases[right_positions] - left_phases[left_positions]

    expected = np.cos(phase_difference[:, None] + omega * offsets[None, :])

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_cross_corr_nb_dot_matches_analytic_sine_solution():
    """
    Over N samples spanning an integer number of periods,

        sum sin(wt + a) sin(w(t + d) + b)
            = N / 2 * cos((b - a) + wd).
    """
    period = 40
    omega = 2 * np.pi / period

    n_samples = period * 12
    padding = period // 2
    total_samples = n_samples + 2 * padding

    left_phase = 0.3
    right_phase = -0.6

    left = _sine(
        total_samples,
        period,
        phase=left_phase,
    )
    right = _sine(
        total_samples,
        period,
        phase=right_phase,
    )

    offsets = np.array(
        [-20, -10, 0, 10, 20],
        dtype=np.intp,
    )

    packed = _packed(
        left,
        right,
        offsets,
        left_bounds=(
            np.array(
                [padding],
                dtype=np.intp,
            ),
            np.array(
                [padding + n_samples],
                dtype=np.intp,
            ),
        ),
    )

    actual = xcorr._cross_corr_nb(
        packed,
        _metric('dot'),
    )

    expected = (n_samples / 2 * np.cos((right_phase - left_phase) + omega * offsets))[
        None, :
    ]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_cross_corr_nb_pearson_is_nan_when_undefined():
    """Pearson is undefined for constant signals and absent overlap."""
    constant = np.ones(
        32,
        dtype=float,
    )

    packed_constant = _packed(
        constant,
        constant,
        np.array(
            [0],
            dtype=np.intp,
        ),
    )

    constant_result = xcorr._cross_corr_nb(
        packed_constant,
        _metric('pearson'),
    )

    assert np.isnan(constant_result[0, 0])

    signal = np.arange(
        32,
        dtype=float,
    )

    packed_no_overlap = _packed(
        signal,
        signal,
        np.array(
            [100],
            dtype=np.intp,
        ),
    )

    no_overlap_result = xcorr._cross_corr_nb(
        packed_no_overlap,
        _metric('pearson'),
    )

    assert np.isnan(no_overlap_result[0, 0])


def test_cross_corr_rolling_nb_matches_analytic_sine_phase():
    """
    A rolling window spanning an integer number of periods has the same
    analytical phase correlation regardless of its anchor position.
    """
    period = 32
    omega = 2 * np.pi / period

    total_samples = 512
    window_samples = period * 4

    left_phase = 0.35
    right_phase = -0.2

    left = _sine(
        total_samples,
        period,
        phase=left_phase,
    )
    right = _sine(
        total_samples,
        period,
        phase=right_phase,
    )

    offsets = np.array(
        [-8, 0, 8],
        dtype=np.intp,
    )

    packed = _packed(
        left,
        right,
        offsets,
    )

    rolling = xcorr._RollingCore(
        window_start=0,
        window_stop=window_samples,
        anchor_start=32,
        anchor_step=7,
        n_times=20,
    )

    actual = xcorr._cross_corr_rolling_nb(
        packed,
        rolling,
        _metric('pearson'),
    )

    expected_per_lag = np.cos((right_phase - left_phase) + omega * offsets)

    expected = np.broadcast_to(
        expected_per_lag[
            None,
            :,
            None,
        ],
        actual.shape,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_cross_corr_rolling_nb_respects_finite_bounds():
    """
    Rolling correlation requires the complete requested window.

    Edge-missing data are represented to the numerical core by finite bounds;
    windows crossing those bounds must return NaN rather than being shortened.
    """
    signal = np.arange(
        32,
        dtype=float,
    )

    packed = _packed(
        signal,
        signal,
        np.array(
            [0],
            dtype=np.intp,
        ),
        left_bounds=(
            np.array(
                [5],
                dtype=np.intp,
            ),
            np.array(
                [25],
                dtype=np.intp,
            ),
        ),
        right_bounds=(
            np.array(
                [5],
                dtype=np.intp,
            ),
            np.array(
                [25],
                dtype=np.intp,
            ),
        ),
    )

    rolling = xcorr._RollingCore(
        window_start=0,
        window_stop=8,
        anchor_start=0,
        anchor_step=4,
        n_times=6,
    )

    actual = xcorr._cross_corr_rolling_nb(
        packed,
        rolling,
        _metric('pearson'),
    )

    expected = np.array(
        [
            [
                [
                    np.nan,
                    np.nan,
                    1.0,
                    1.0,
                    1.0,
                    np.nan,
                ]
            ]
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
        equal_nan=True,
    )


def test_cross_corr_rolling_nb_weighted_dot_kernel():
    """
    A kernel is an unnormalized sample-wise weight.

    For constant signals x=a and y=b,

        weighted dot = a * b * sum(kernel).
    """
    left_value = 2.0
    right_value = 3.0

    left = np.full(
        64,
        left_value,
    )
    right = np.full(
        64,
        right_value,
    )

    offsets = np.array(
        [-2, 0, 3],
        dtype=np.intp,
    )

    packed = _packed(
        left,
        right,
        offsets,
    )

    rolling = xcorr._RollingCore(
        window_start=-2,
        window_stop=3,
        anchor_start=10,
        anchor_step=4,
        n_times=6,
    )

    kernel = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]
    )

    actual = xcorr._cross_corr_rolling_nb(
        packed,
        rolling,
        _metric(
            'dot',
            kernel=kernel,
        ),
    )

    expected_value = left_value * right_value * kernel.sum()

    expected = np.full(
        actual.shape,
        expected_value,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )

    # A unit kernel must reduce exactly to the ordinary dot product.
    weighted_unit = xcorr._cross_corr_rolling_nb(
        packed,
        rolling,
        _metric(
            'dot',
            kernel=np.ones(len(kernel)),
        ),
    )

    unweighted = xcorr._cross_corr_rolling_nb(
        packed,
        rolling,
        _metric('dot'),
    )

    np.testing.assert_allclose(
        weighted_unit,
        unweighted,
        rtol=RTOL,
        atol=ATOL,
    )
