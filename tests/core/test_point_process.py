import numpy as np
import pytest

import nocte.core._point_process as point_process


def test_as_times_1d_accepts_scalar_and_returns_float_array():
    result = point_process.as_times_1d(3)

    assert result.dtype == float
    assert result.shape == (1,)
    np.testing.assert_array_equal(result, [3.0])


def test_as_times_1d_rejects_non_1d_and_nonfinite_values():
    with pytest.raises(ValueError, match='one-dimensional'):
        point_process.as_times_1d([[1.0, 2.0]])

    with pytest.raises(ValueError, match='finite'):
        point_process.as_times_1d([1.0, np.nan])

    with pytest.raises(ValueError, match='finite'):
        point_process.as_times_1d([1.0, np.inf])


def test_as_times_1d_copy_controls_memory_sharing():
    source = np.array([1.0, 2.0, 3.0])

    shared = point_process.as_times_1d(source, copy=False)
    copied = point_process.as_times_1d(source, copy=True)

    assert np.shares_memory(shared, source)
    assert not np.shares_memory(copied, source)


def test_as_sorted_times_allows_duplicates_but_rejects_decreasing_times():
    result = point_process.as_sorted_times_1d([1.0, 1.0, 2.0])
    np.testing.assert_array_equal(result, [1.0, 1.0, 2.0])

    with pytest.raises(ValueError, match='monotonically non-decreasing'):
        point_process.as_sorted_times_1d([1.0, 3.0, 2.0])


def test_as_bin_edges_requires_at_least_two_strictly_increasing_edges():
    with pytest.raises(ValueError, match='at least two'):
        point_process.as_bin_edges([0.0])

    with pytest.raises(ValueError, match='strictly increasing'):
        point_process.as_bin_edges([0.0, 1.0, 1.0])

    with pytest.raises(ValueError, match='strictly increasing'):
        point_process.as_bin_edges([0.0, 2.0, 1.0])


def test_sample_centers_respects_margin_and_regular_step():
    result = point_process.sample_centers(0.0, 10.0, 2.0, margin=1.0)

    np.testing.assert_array_equal(result, [1.0, 3.0, 5.0, 7.0, 9.0])


def test_sample_centers_returns_empty_when_margin_does_not_fit():
    result = point_process.sample_centers(0.0, 10.0, 1.0, margin=6.0)

    assert result.dtype == float
    assert result.shape == (0,)


def test_sample_centers_rejects_invalid_geometry():
    with pytest.raises(ValueError, match='greater than or equal'):
        point_process.sample_centers(2.0, 1.0, 1.0)

    with pytest.raises(ValueError, match='positive'):
        point_process.sample_centers(0.0, 1.0, 0.0)

    with pytest.raises(ValueError, match='non-negative'):
        point_process.sample_centers(0.0, 1.0, 1.0, margin=-1.0)


def test_count_between_many_is_half_open_and_handles_duplicates_and_empty_trains():
    trains = (
        np.array([0.0, 1.0, 1.0, 2.0, 3.0]),
        np.array([], dtype=float),
    )

    result = point_process.count_between_many(trains, 1.0, 3.0)

    np.testing.assert_array_equal(result, [3, 0])


def test_count_between_many_empty_interval_counts_nothing():
    trains = (np.array([1.0, 1.0]),)

    result = point_process.count_between_many(trains, 1.0, 1.0)

    np.testing.assert_array_equal(result, [0])


def test_count_bins_many_is_train_major_and_final_edge_is_exclusive():
    trains = (
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([0.5, 2.5]),
    )
    edges = np.array([0.0, 1.0, 2.0, 3.0])

    result = point_process.count_bins_many(trains, edges)

    assert result.shape == (2, 3)
    np.testing.assert_array_equal(
        result,
        [
            [1, 1, 1],
            [1, 0, 1],
        ],
    )


def test_count_bins_many_preserves_empty_train_axis():
    result = point_process.count_bins_many((), np.array([0.0, 1.0, 2.0]))

    assert result.shape == (0, 2)


def test_count_rolling_many_uses_centered_half_open_windows():
    trains = (np.array([0.0, 1.0, 2.0, 3.0]),)
    sample_times = np.array([1.0, 2.0])

    result = point_process.count_rolling_many(trains, sample_times, window=2.0)

    np.testing.assert_array_equal(result, [[2, 2]])


def test_count_rolling_many_rejects_nonpositive_window():
    with pytest.raises(ValueError, match='positive'):
        point_process.count_rolling_many((np.array([1.0]),), [1.0], window=0.0)


def test_gaussian_rate_many_matches_single_spike_analytical_peak():
    sigma = 2.0
    trains = (np.array([5.0]),)

    result = point_process.gaussian_rate_many(
        trains,
        np.array([5.0]),
        sigma=sigma,
        width=5.0,
    )

    expected = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(expected)


def test_gaussian_rate_many_is_additive_for_coincident_events():
    sigma = 1.5
    trains = (np.array([5.0, 5.0]),)

    result = point_process.gaussian_rate_many(
        trains,
        np.array([5.0]),
        sigma=sigma,
        width=5.0,
    )

    expected_single = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    assert result[0, 0] == pytest.approx(2.0 * expected_single)


def test_gaussian_rate_many_returns_zero_for_empty_train():
    result = point_process.gaussian_rate_many(
        (np.array([], dtype=float),),
        np.array([0.0, 1.0]),
        sigma=1.0,
    )

    np.testing.assert_array_equal(result, [[0.0, 0.0]])


def test_gaussian_rate_many_rejects_invalid_kernel_parameters():
    train = (np.array([0.0]),)

    with pytest.raises(ValueError, match='sigma'):
        point_process.gaussian_rate_many(train, [0.0], sigma=0.0)

    with pytest.raises(ValueError, match='width'):
        point_process.gaussian_rate_many(train, [0.0], sigma=1.0, width=0.0)
