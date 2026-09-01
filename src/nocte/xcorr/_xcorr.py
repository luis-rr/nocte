"""Continuous full-signal and rolling cross-correlation for Traces.

Positive lag compares ``left(t)`` with ``right(t + lag)``. A positive-lag
peak therefore means that the left signal leads the right signal.
"""

from __future__ import annotations

import typing

import numpy as np

import nocte._coll.traces
import nocte._coll.windows
import nocte._core.matching
from nocte.xcorr import _core

CorrelationMethod = typing.Literal['pearson', 'dot']


def cross_corr(
    left: nocte._coll.traces.Traces,
    right: nocte._coll.traces.Traces,
    matches: nocte._core.matching.Matches,
    *,
    lags: np.ndarray,
    method: CorrelationMethod = 'pearson',
) -> nocte._coll.traces.Traces:
    """
    Cross-correlate matched traces over their available finite overlap.

    Positive lag compares ``left(t)`` with ``right(t + lag)``. Lags are in
    milliseconds and must form a regular grid aligned to integer source
    samples. Leading and trailing NaNs reduce overlap; internal gaps are
    rejected.

    The result contains one trace per match, with lag as its sampled temporal
    coordinate.
    """
    return _core.cross_corr(
        left,
        right,
        matches,
        lags=lags,
        method=method,
    )


def auto_corr(
    traces: nocte._coll.traces.Traces,
    *,
    lags: np.ndarray,
    method: CorrelationMethod = 'pearson',
) -> nocte._coll.traces.Traces:
    """
    Autocorrelate every trace over its available finite overlap.

    This is the identity-match specialization of ``cross_corr``. Trace
    identities and metadata are preserved.
    """
    return _core.auto_corr(
        traces,
        lags=lags,
        method=method,
    )


def cross_corr_rolling(
    left: nocte._coll.traces.Traces,
    right: nocte._coll.traces.Traces,
    matches: nocte._core.matching.Matches,
    *,
    lags: np.ndarray,
    window: nocte._coll.windows.Win,
    step: float,
    method: CorrelationMethod = 'pearson',
    kernel: np.ndarray | None = None,
) -> nocte._coll.traces.Traces:
    """
    Cross-correlate matched traces through fixed windows in time.

    For output time ``t`` and lag ``d``, the calculation compares ``t + window``
    on the left with ``t + window + d`` on the right. The result contains one
    trace per ``(match, lag)`` pair; rolling time is the sampled coordinate and
    lag is item metadata.

    Windows are never shortened around leading or trailing NaNs. If either
    requested segment is not fully observed, that result is NaN. ``kernel``
    supplies unnormalized sample-wise weights for ``method='dot'`` and must
    contain exactly one value per sample in ``window``.
    """
    return _core.cross_corr_rolling(
        left,
        right,
        matches,
        lags=lags,
        window=window,
        step=step,
        method=method,
        kernel=kernel,
    )


def auto_corr_rolling(
    traces: nocte._coll.traces.Traces,
    *,
    lags: np.ndarray,
    window: nocte._coll.windows.Win,
    step: float,
    method: CorrelationMethod = 'pearson',
    kernel: np.ndarray | None = None,
) -> nocte._coll.traces.Traces:
    """
    Rolling autocorrelation for every trace and requested lag.

    The result contains one trace per ``(source trace, lag)`` pair.
    """
    return _core.auto_corr_rolling(
        traces,
        lags=lags,
        window=window,
        step=step,
        method=method,
        kernel=kernel,
    )
