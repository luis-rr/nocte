"""Low-level numerical helpers for one or more point processes."""

from __future__ import annotations

import collections.abc

import numba as nb
import numpy as np
import pandas as pd

TimeArrayLike = float | collections.abc.Sequence[float] | np.ndarray | pd.Index
SortedTimes = collections.abc.Sequence[np.ndarray]


def as_times_1d(
    times: TimeArrayLike,
    *,
    copy: bool = False,
) -> np.ndarray:
    """Return finite one-dimensional float timestamps."""
    if copy:
        values = np.array(times, dtype=float, order='C', copy=True)
    else:
        values = np.asarray(times, dtype=float, order='C')

    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim != 1:
        raise ValueError('Times must be scalar or one-dimensional')

    if not np.isfinite(values).all():
        raise ValueError('Times must be finite')

    return values


def as_sorted_times_1d(
    times: TimeArrayLike,
    *,
    copy: bool = False,
) -> np.ndarray:
    """Return finite, monotonically non-decreasing one-dimensional timestamps."""
    values = as_times_1d(times, copy=copy)

    if len(values) >= 2 and np.any(values[1:] < values[:-1]):
        raise ValueError('Times must be monotonically non-decreasing')

    return values


def as_bin_edges(bins: TimeArrayLike) -> np.ndarray:
    """Return validated strictly increasing bin edges."""
    edges = as_times_1d(bins)

    if len(edges) < 2:
        raise ValueError('Bins must contain at least two edges')
    if np.any(np.diff(edges) <= 0):
        raise ValueError('Bins must be strictly increasing')

    return edges


def sample_centers(
    start: float,
    stop: float,
    step: float,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Return regular centers whose symmetric margin lies inside ``[start, stop)``."""
    start = float(start)
    stop = float(stop)
    step = float(step)
    margin = float(margin)

    if not np.all(np.isfinite([start, stop, step, margin])):
        raise ValueError('Sampling bounds, step, and margin must be finite')
    if stop < start:
        raise ValueError('Sampling stop must be greater than or equal to start')
    if step <= 0:
        raise ValueError('Sampling step must be positive')
    if margin < 0:
        raise ValueError('Sampling margin must be non-negative')

    first = start + margin
    last = stop - margin
    if last < first:
        return np.empty(0, dtype=float)

    span = last - first
    n = int(np.floor(span / step + 1e-12)) + 1
    return first + np.arange(n, dtype=float) * step


def count_between_many(
    trains: SortedTimes,
    start: float,
    stop: float,
) -> np.ndarray:
    """Count each sorted point process in the half-open interval ``[start, stop)``."""
    start = float(start)
    stop = float(stop)

    if not np.all(np.isfinite([start, stop])):
        raise ValueError('Count bounds must be finite')
    if stop < start:
        raise ValueError('Count stop must be greater than or equal to start')

    result = np.empty(len(trains), dtype=np.int64)

    for i, times in enumerate(trains):
        left = np.searchsorted(times, start, side='left')
        right = np.searchsorted(times, stop, side='left')
        result[i] = right - left

    return result


def count_bins_many(
    trains: SortedTimes,
    edges: np.ndarray,
) -> np.ndarray:
    """Count each sorted point process in common half-open bins.

    Returns an array shaped ``(n_trains, n_bins)``.
    """
    edges = as_bin_edges(edges)
    result = np.empty((len(trains), len(edges) - 1), dtype=np.int64)

    for i, times in enumerate(trains):
        left = np.searchsorted(times, edges[:-1], side='left')
        right = np.searchsorted(times, edges[1:], side='left')
        result[i] = right - left

    return result


def count_rolling_many(
    trains: SortedTimes,
    sample_times: np.ndarray,
    window: float,
) -> np.ndarray:
    """Count each sorted point process in centered half-open windows.

    Returns an array shaped ``(n_trains, n_samples)``.
    """
    sample_times = as_times_1d(sample_times)
    window = float(window)

    if not np.isfinite(window) or window <= 0:
        raise ValueError('window must be finite and positive')

    half_window = window * 0.5
    starts = sample_times - half_window
    stops = sample_times + half_window
    result = np.empty((len(trains), len(sample_times)), dtype=np.int64)

    for i, times in enumerate(trains):
        left = np.searchsorted(times, starts, side='left')
        right = np.searchsorted(times, stops, side='left')
        result[i] = right - left

    return result


@nb.njit(parallel=True)
def _gaussian_rate_sorted_nb(
    times: np.ndarray,
    sample_times: np.ndarray,
    sigma: float,
    width: float,
) -> np.ndarray:
    """Evaluate a truncated Gaussian kernel rate on one sorted point process."""
    result = np.empty(len(sample_times), dtype=np.float64)
    half_width = sigma * width
    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    for i in nb.prange(len(sample_times)):
        center = sample_times[i]
        left = np.searchsorted(times, center - half_width, side='left')
        right = np.searchsorted(times, center + half_width, side='left')

        acc = 0.0
        for j in range(left, right):
            u = (center - times[j]) / sigma
            acc += norm * np.exp(-0.5 * u * u)

        result[i] = acc

    return result


def gaussian_rate_many(
    trains: SortedTimes,
    sample_times: np.ndarray,
    sigma: float,
    width: float = 5.0,
) -> np.ndarray:
    """Evaluate Gaussian kernel rates for one or more sorted point processes.

    The expensive single-process calculation remains Numba compiled. The outer
    loop intentionally stays simple: with the expected hundreds of trains it is
    cheap orchestration around a known-hot numerical kernel. Returns an array
    shaped ``(n_trains, n_samples)`` in inverse timestamp units.
    """
    sample_times = as_times_1d(sample_times)
    sigma = float(sigma)
    width = float(width)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('sigma must be finite and positive')
    if not np.isfinite(width) or width <= 0:
        raise ValueError('width must be finite and positive')

    result = np.empty((len(trains), len(sample_times)), dtype=np.float64)

    for i, times in enumerate(trains):
        result[i] = _gaussian_rate_sorted_nb(times, sample_times, sigma, width)

    return result
