from __future__ import annotations

import numpy as np

import nocte._coll.traces
from nocte._core import num
from nocte.spec import _core

RTOL = 1e-11
ATOL = 1e-11


def _traces(
    values: np.ndarray,
    *,
    hz: float,
) -> nocte._coll.traces.Traces:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim == 1:
        values = values[None, :]

    return nocte._coll.traces.Traces.from_array(
        values,
        hz,
        start=0.0,
    )


def _cosine(
    n_samples: int,
    *,
    hz: float,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> np.ndarray:
    samples = np.arange(
        n_samples,
        dtype=np.float64,
    )

    return amplitude * np.cos(2.0 * np.pi * frequency * samples / hz + phase)


def _hann_cosine_psd(
    *,
    hz: float,
    nperseg: int,
    frequency: float,
    amplitude: float,
) -> np.ndarray:
    """
    Closed-form one-sided Hann-window PSD of a bin-centred cosine.

    For a cosine of amplitude A exactly on FFT bin k, the periodic Hann
    window places all power in bins k - 1, k, k + 1:

        P[k]     = A² / (3 df)
        P[k ± 1] = A² / (12 df)

    Their integrated power is A² / 2, the mean square of the cosine.
    """
    df = hz / nperseg
    bin_ = frequency / df
    rounded = round(bin_)

    if not np.isclose(
        bin_,
        rounded,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError('frequency must lie exactly on an FFT bin')

    index = int(rounded)

    if index <= 0 or index >= nperseg // 2:
        raise ValueError('frequency must be away from DC and Nyquist')

    power = np.zeros(
        nperseg // 2 + 1,
        dtype=np.float64,
    )

    power[index] = amplitude**2 / (3.0 * df)
    power[index - 1] = amplitude**2 / (12.0 * df)
    power[index + 1] = amplitude**2 / (12.0 * df)

    return power


def test_analytic_signal_exact_complex_exponential():
    """
    Phase, amplitude, and instantaneous frequency have closed-form solutions
    for an exact complex exponential.
    """
    hz = 200.0
    frequency = 25.0
    phase = 0.37
    n_samples = 128

    samples = np.arange(
        n_samples,
        dtype=np.float64,
    )
    expected_phase = phase + 2.0 * np.pi * frequency * samples / hz

    values = np.exp(1j * expected_phase)[None, :]

    analytic = _core.Analytic(
        values=np.ascontiguousarray(
            values,
            dtype=np.complex128,
        ),
        bounds=num.Bounds(
            first=np.array(
                [0],
                dtype=np.intp,
            ),
            stop=np.array(
                [n_samples],
                dtype=np.intp,
            ),
        ),
    )

    amplitude = analytic.amplitude()
    unwrapped_phase = analytic.phase(
        unwrap=True,
    )
    actual_frequency = analytic.frequency(hz)

    np.testing.assert_allclose(
        amplitude,
        1.0,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        unwrapped_phase[0],
        expected_phase,
        rtol=RTOL,
        atol=ATOL,
    )

    assert np.isnan(actual_frequency[0, 0])

    np.testing.assert_allclose(
        actual_frequency[0, 1:],
        frequency,
        rtol=RTOL,
        atol=ATOL,
    )


def test_hilbert_of_cosine_is_complex_exponential_with_finite_support():
    """
    For an integer-period cosine,

        cos(phi) + i H[cos(phi)] = exp(i phi).

    Edge NaNs are outside the finite support and must remain NaN.
    """
    hz = 256.0
    frequency = 16.0
    phase = -0.43

    padding = 64
    finite_samples = 512
    n_samples = finite_samples + 2 * padding

    finite = _cosine(
        finite_samples,
        hz=hz,
        frequency=frequency,
        phase=phase,
    )

    values = np.full(
        n_samples,
        np.nan,
        dtype=np.float64,
    )
    values[padding : padding + finite_samples] = finite

    traces = _traces(
        values,
        hz=hz,
    )

    analytic = _core.Analytic.from_traces(traces)

    finite_positions = np.arange(
        finite_samples,
        dtype=np.float64,
    )
    expected_phase = phase + 2.0 * np.pi * frequency * finite_positions / hz
    expected = np.exp(1j * expected_phase)

    assert np.isnan(
        analytic.values[
            0,
            :padding,
        ]
    ).all()
    assert np.isnan(
        analytic.values[
            0,
            padding + finite_samples :,
        ]
    ).all()

    np.testing.assert_allclose(
        analytic.values[
            0,
            padding : padding + finite_samples,
        ],
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_butterworth_zero_phase_gain_is_half_at_cutoff():
    """
    A Butterworth filter has magnitude 1 / sqrt(2) at its cutoff.

    Forward-backward filtering applies the magnitude response twice, so the
    resulting zero-phase amplitude at the cutoff is exactly one half.
    """
    hz = 256.0
    n_samples = 8192
    phase = 0.37
    order = 4

    cases = [
        (
            _core.Butterworth.low_pass(
                hz=hz,
                cutoff=32.0,
                order=order,
            ),
            32.0,
        ),
        (
            _core.Butterworth.high_pass(
                hz=hz,
                cutoff=32.0,
                order=order,
            ),
            32.0,
        ),
        (
            _core.Butterworth.band_pass(
                hz=hz,
                band=(
                    24.0,
                    48.0,
                ),
                order=order,
            ),
            24.0,
        ),
        (
            _core.Butterworth.band_pass(
                hz=hz,
                band=(
                    24.0,
                    48.0,
                ),
                order=order,
            ),
            48.0,
        ),
    ]

    interior = slice(
        1024,
        -1024,
    )

    for filter_, frequency in cases:
        signal = _cosine(
            n_samples,
            hz=hz,
            frequency=frequency,
            phase=phase,
        )

        traces = _traces(
            signal,
            hz=hz,
        )

        actual = filter_.apply(traces).values[0]

        expected = 0.5 * signal

        np.testing.assert_allclose(
            actual[interior],
            expected[interior],
            rtol=1e-10,
            atol=1e-10,
        )


def test_welch_bin_centred_cosine_matches_closed_form_hann_psd():
    """
    A bin-centred cosine under a periodic Hann window has an exact three-bin
    one-sided PSD.
    """
    hz = 256.0
    nperseg = 256
    n_samples = 1024

    frequency = 32.0
    amplitude = 2.5
    phase = 0.31

    signal = _cosine(
        n_samples,
        hz=hz,
        frequency=frequency,
        amplitude=amplitude,
        phase=phase,
    )

    traces = _traces(
        signal,
        hz=hz,
    )

    estimate = _core.Welch.from_traces(
        traces,
        segment=1000.0,
    )

    assert estimate.nperseg == nperseg

    actual = estimate.power(traces.values)

    expected = _hann_cosine_psd(
        hz=hz,
        nperseg=nperseg,
        frequency=frequency,
        amplitude=amplitude,
    )

    np.testing.assert_allclose(
        actual[0],
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_band_power_of_bin_centred_cosine_equals_mean_square():
    """
    The three Hann bins around a bin-centred cosine integrate exactly to

        A² / 2,

    which is the cosine's mean square.
    """
    hz = 256.0
    nperseg = 256

    frequency = 32.0
    amplitude = 2.5

    frequency_grid = np.fft.rfftfreq(
        nperseg,
        d=1.0 / hz,
    )

    estimate = _core.Welch(
        hz=hz,
        nperseg=nperseg,
        frequency=np.asarray(
            frequency_grid,
            dtype=np.float64,
        ),
    )

    plan = _core.BandPlan.from_bands(
        {
            'tone': (
                31.0,
                33.0,
            )
        },
        estimate,
    )

    power = _hann_cosine_psd(
        hz=hz,
        nperseg=nperseg,
        frequency=frequency,
        amplitude=amplitude,
    )[None, :]

    actual = plan.integrate(power)

    expected = np.array(
        [
            [
                amplitude**2 / 2.0,
            ]
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_rolling_welch_of_stationary_cosine_matches_closed_form_at_every_time():
    """
    A stationary bin-centred cosine has the same PSD in every rolling window,
    even when successive windows begin at different phases of the cosine.

    More than 32 windows are used so the numerical batching boundary is also
    exercised.
    """
    hz = 256.0
    n_samples = 512

    window_samples = 128
    step_samples = 7

    frequency = 32.0
    amplitude = 1.7
    phase = -0.28

    signal = _cosine(
        n_samples,
        hz=hz,
        frequency=frequency,
        amplitude=amplitude,
        phase=phase,
    )

    traces = _traces(
        signal,
        hz=hz,
    )

    estimate = _core.Welch.from_traces(
        traces,
        segment=500.0,
    )

    rolling = _core.Rolling.from_traces(
        traces,
        window=500.0,
        step=1000.0 * step_samples / hz,
    )

    assert estimate.nperseg == window_samples
    assert rolling.window_samples == window_samples
    assert rolling.step_samples == step_samples
    assert rolling.grid.n_samples > 32

    actual = rolling.welch(
        traces,
        estimate,
    )

    expected_per_time = _hann_cosine_psd(
        hz=hz,
        nperseg=window_samples,
        frequency=frequency,
        amplitude=amplitude,
    )

    expected = np.broadcast_to(
        expected_per_time[
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


def test_rolling_band_power_of_stationary_cosine_equals_mean_square():
    """
    Rolling band power of a stationary bin-centred cosine is constant and
    equal to its mean square, A² / 2.
    """
    hz = 256.0
    n_samples = 512

    window_samples = 128
    step_samples = 7

    frequency = 32.0
    amplitude = 1.7
    phase = 0.61

    signal = _cosine(
        n_samples,
        hz=hz,
        frequency=frequency,
        amplitude=amplitude,
        phase=phase,
    )

    traces = _traces(
        signal,
        hz=hz,
    )

    estimate = _core.Welch.from_traces(
        traces,
        segment=500.0,
    )

    df = hz / window_samples

    plan = _core.BandPlan.from_bands(
        {
            'tone': (
                frequency - df,
                frequency + df,
            )
        },
        estimate,
    )

    rolling = _core.Rolling.from_traces(
        traces,
        window=500.0,
        step=1000.0 * step_samples / hz,
    )

    actual = rolling.band_power(
        traces,
        estimate,
        plan,
    )

    expected = np.full(
        actual.shape,
        amplitude**2 / 2.0,
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=RTOL,
        atol=ATOL,
    )
