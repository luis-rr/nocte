"""Filtering, Hilbert transforms, and power spectral analysis for Traces."""

from __future__ import annotations

import collections.abc

import numpy as np
import numpy.typing as npt
import pandas as pd

from nocte._coll.traces import Traces
from nocte.spec import _core

Band = tuple[float, float]
Bands = collections.abc.Mapping[str, Band]


def low_pass(
    traces: Traces,
    cutoff: float,
    *,
    order: int = 2,
) -> Traces:
    """Zero-phase Butterworth low-pass filtering."""
    return _core.low_pass(
        traces,
        cutoff,
        order=order,
    )


def high_pass(
    traces: Traces,
    cutoff: float,
    *,
    order: int = 2,
) -> Traces:
    """Zero-phase Butterworth high-pass filtering."""
    return _core.high_pass(
        traces,
        cutoff,
        order=order,
    )


def band_pass(
    traces: Traces,
    band: Band,
    *,
    order: int = 2,
) -> Traces:
    """Zero-phase Butterworth band-pass filtering."""
    return _core.band_pass(
        traces,
        band,
        order=order,
    )


def hilbert(
    traces: Traces,
) -> npt.NDArray[np.complex128]:
    """
    Return the complex analytic signal for every trace.

    The returned array has the same shape as ``traces.values``. Leading and
    trailing missing samples remain missing; internal gaps are rejected.
    """
    return _core.hilbert(traces)


def hilbert_phase(
    traces: Traces,
    *,
    unwrap: bool = False,
) -> Traces:
    """Return Hilbert phase in radians, optionally unwrapped in time."""
    return _core.hilbert_phase(
        traces,
        unwrap=unwrap,
    )


def hilbert_amplitude(
    traces: Traces,
) -> Traces:
    """Return the magnitude of each trace's complex analytic signal."""
    return _core.hilbert_amplitude(traces)


def instantaneous_frequency(
    traces: Traces,
) -> Traces:
    """
    Return instantaneous frequency in Hz from unwrapped Hilbert phase.

    The first finite sample of each trace is missing because frequency is
    estimated from successive phase differences.
    """
    return _core.instantaneous_frequency(traces)


def welch(
    traces: Traces,
    *,
    segment: float = 4_000.0,
    db: bool = False,
) -> pd.DataFrame:
    """
    Estimate one Welch power spectral density per trace.

    ``segment`` is the Welch segment duration in milliseconds. Rows preserve
    source trace identity and columns are frequencies in Hz.
    """
    return _core.welch(
        traces,
        segment=segment,
        db=db,
    )


def band_power(
    traces: Traces,
    bands: Bands,
    *,
    segment: float = 4_000.0,
    db: bool = False,
) -> pd.DataFrame:
    """Integrate Welch power spectral density within named frequency bands."""
    return _core.band_power(
        traces,
        bands,
        segment=segment,
        db=db,
    )


def welch_rolling(
    traces: Traces,
    *,
    window: float = 10_000.0,
    step: float = 1_000.0,
    segment: float = 4_000.0,
    db: bool = False,
) -> Traces:
    """
    Estimate Welch power spectral density through fixed sliding windows.

    The result contains one trace per ``(source trace, frequency)`` pair.
    Rolling-window centers form the result time coordinate. A rolling window
    touching missing source samples produces missing power values.
    """
    return _core.welch_rolling(
        traces,
        window=window,
        step=step,
        segment=segment,
        db=db,
    )


def band_power_rolling(
    traces: Traces,
    bands: Bands,
    *,
    window: float = 10_000.0,
    step: float = 1_000.0,
    segment: float = 4_000.0,
    db: bool = False,
) -> Traces:
    """
    Integrate Welch power within named bands through fixed sliding windows.

    The result contains one trace per ``(source trace, band)`` pair.
    Rolling-window centers form the result time coordinate. The complete
    rolling frequency cube is never materialized.
    """
    return _core.band_power_rolling(
        traces,
        bands,
        window=window,
        step=step,
        segment=segment,
        db=db,
    )
