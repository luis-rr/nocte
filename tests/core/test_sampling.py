import logging

import numpy as np
import pytest

import nocte._core.sampling


def test_sampling_rate():
    sr = nocte._core.sampling.SamplingRate(30_000)

    assert sr.rate == 30_000.0
    assert isinstance(sr.rate, float)


@pytest.mark.parametrize(
    'rate',
    [
        0,
        -1,
        np.inf,
        np.nan,
    ],
)
def test_sampling_rate_rejects_invalid_rate(rate):
    with pytest.raises(ValueError):
        nocte._core.sampling.SamplingRate(rate)


def test_from_period_ms():
    sr = nocte._core.sampling.SamplingRate.from_period_ms(0.5)

    assert sr.rate == 2000.0


@pytest.mark.parametrize(
    'period_ms',
    [
        0,
        -1,
        np.inf,
        np.nan,
    ],
)
def test_from_period_ms_rejects_invalid_period(period_ms):
    with pytest.raises(ValueError):
        nocte._core.sampling.SamplingRate.from_period_ms(period_ms)


def test_period_ms():
    sr = nocte._core.sampling.SamplingRate(2000)

    assert sr.period_ms == 0.5


@pytest.mark.parametrize(
    ('target_hz', 'expected_stride'),
    [
        (30_000, 1),
        (15_000, 2),
        (1000, 30),
        (1200, 25),
    ],
)
def test_stride_for(target_hz, expected_stride):
    sr = nocte._core.sampling.SamplingRate(30_000)

    assert sr.stride_for(target_hz) == expected_stride


@pytest.mark.parametrize(
    'target_hz',
    [
        0,
        -1,
        np.inf,
        np.nan,
        30_001,
    ],
)
def test_stride_for_rejects_invalid_target_rate(target_hz):
    sr = nocte._core.sampling.SamplingRate(30_000)

    with pytest.raises(ValueError):
        sr.stride_for(target_hz)


def test_match_hz_exact():
    sr = nocte._core.sampling.SamplingRate(30_000)

    assert sr.match_hz(1000) == 1000.0


def test_match_hz_returns_nearest_integer_stride():
    sr = nocte._core.sampling.SamplingRate(30_000)

    matched = sr.match_hz(1100)

    expected_stride = 27
    assert matched == pytest.approx(30_000 / expected_stride)


def test_match_hz_warns_when_adjusted(caplog):
    sr = nocte._core.sampling.SamplingRate(30_000)

    with caplog.at_level(logging.WARNING):
        sr.match_hz(1100)

    assert 'Adjusting target sampling rate' in caplog.text


def test_check_stride():
    sr = nocte._core.sampling.SamplingRate(30_000)

    assert sr.check_stride(1000)
    assert sr.check_stride(1200)
    assert not sr.check_stride(1100)


def test_assert_stride():
    sr = nocte._core.sampling.SamplingRate(30_000)

    sr.assert_stride(1000)

    with pytest.raises(AssertionError):
        sr.assert_stride(1100)


def test_ms_to_samples_scalar():
    sr = nocte._core.sampling.SamplingRate(2000)

    samples = sr.ms_to_samples(1.5)

    assert samples == 3
    assert isinstance(samples, int)


def test_ms_to_samples_array():
    sr = nocte._core.sampling.SamplingRate(2000)

    time_ms = np.array([0.0, 0.5, 1.0, 2.0])
    samples = sr.ms_to_samples(time_ms)

    np.testing.assert_array_equal(
        samples,
        np.array([0, 1, 2, 4]),
    )

    assert np.issubdtype(samples.dtype, np.integer)


def test_samples_to_ms_scalar():
    sr = nocte._core.sampling.SamplingRate(2000)

    time_ms = sr.samples_to_ms(3)

    assert time_ms == 1.5
    assert isinstance(time_ms, float)


def test_samples_to_ms_array():
    sr = nocte._core.sampling.SamplingRate(2000)

    samples = np.array([0, 1, 2, 4])
    time_ms = sr.samples_to_ms(samples)

    np.testing.assert_allclose(
        time_ms,
        np.array([0.0, 0.5, 1.0, 2.0]),
    )


def test_sample_time_roundtrip():
    sr = nocte._core.sampling.SamplingRate(30_000)

    samples = np.array([0, 1, 2, 100, 10_000])

    reconstructed = sr.ms_to_samples(sr.samples_to_ms(samples))

    np.testing.assert_array_equal(reconstructed, samples)


@pytest.mark.parametrize(
    ('value_ms', 'expected'),
    [
        (1.24, 1.0),
        (1.26, 1.5),
        (-1.24, -1.0),
        (-1.26, -1.5),
    ],
)
def test_round_to_period(value_ms, expected):
    sr = nocte._core.sampling.SamplingRate(2000)

    assert sr.round_to_period(value_ms) == expected


def test_round_to_period_warns_with_description(caplog):
    sr = nocte._core.sampling.SamplingRate(1000)

    with caplog.at_level(logging.WARNING):
        result = sr.round_to_period(10.4, desc='window length')

    assert result == 10.0
    assert 'window length' in caplog.text


@pytest.mark.parametrize(
    ('start', 'last', 'expected_times'),
    [
        (0.0, 20.0, [0.0, 10.0, 20.0]),
        (0.0, 25.0, [0.0, 10.0, 20.0]),
        (10.0, 5.0, []),
    ],
)
def test_time_grid_from_start_last(start, last, expected_times):
    grid = nocte._core.sampling.TimeGrid.from_start_last(
        sampling=nocte._core.sampling.SamplingRate(100),
        start=start,
        last=last,
    )

    np.testing.assert_allclose(grid.times, expected_times)
