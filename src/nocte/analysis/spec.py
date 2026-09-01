"""Filtering, Hilbert transforms, and power spectral analysis for Traces."""

from __future__ import annotations

import collections.abc

import numpy as np
import numpy.typing as npt
import pandas as pd

import nocte.core.traces
from nocte.analysis import _spec_core

Band = tuple[float, float]
Bands = collections.abc.Mapping[str, Band]


def low_pass(
    traces: nocte.core.traces.Traces,
    cutoff: float,
    *,
    order: int = 2,
) -> nocte.core.traces.Traces:
    """Zero-phase Butterworth low-pass filtering."""
    return _spec_core.low_pass(
        traces,
        cutoff,
        order=order,
    )


def high_pass(
    traces: nocte.core.traces.Traces,
    cutoff: float,
    *,
    order: int = 2,
) -> nocte.core.traces.Traces:
    """Zero-phase Butterworth high-pass filtering."""
    return _spec_core.high_pass(
        traces,
        cutoff,
        order=order,
    )


def band_pass(
    traces: nocte.core.traces.Traces,
    band: Band,
    *,
    order: int = 2,
) -> nocte.core.traces.Traces:
    """Zero-phase Butterworth band-pass filtering."""
    return _spec_core.band_pass(
        traces,
        band,
        order=order,
    )


def hilbert(
    traces: nocte.core.traces.Traces,
) -> npt.NDArray[np.complex128]:
    """
    Return the complex analytic signal for every trace.

    The returned array has the same shape as ``traces.values``. Leading and
    trailing missing samples remain missing; internal gaps are rejected.
    """
    return _spec_core.hilbert(traces)


def hilbert_phase(
    traces: nocte.core.traces.Traces,
    *,
    unwrap: bool = False,
) -> nocte.core.traces.Traces:
    """Return Hilbert phase in radians, optionally unwrapped in time."""
    return _spec_core.hilbert_phase(
        traces,
        unwrap=unwrap,
    )


def hilbert_amplitude(
    traces: nocte.core.traces.Traces,
) -> nocte.core.traces.Traces:
    """Return the magnitude of each trace's complex analytic signal."""
    return _spec_core.hilbert_amplitude(traces)


def instantaneous_frequency(
    traces: nocte.core.traces.Traces,
) -> nocte.core.traces.Traces:
    """
    Return instantaneous frequency in Hz from unwrapped Hilbert phase.

    The first finite sample of each trace is missing because frequency is
    estimated from successive phase differences.
    """
    return _spec_core.instantaneous_frequency(traces)


def welch(
    traces: nocte.core.traces.Traces,
    *,
    segment: float = 4_000.0,
    db: bool = False,
) -> pd.DataFrame:
    """
    Estimate one Welch power spectral density per trace.

    ``segment`` is the Welch segment duration in milliseconds. Rows preserve
    source trace identity and columns are frequencies in Hz.
    """
    return _spec_core.welch(
        traces,
        segment=segment,
        db=db,
    )


def band_power(
    traces: nocte.core.traces.Traces,
    bands: Bands,
    *,
    segment: float = 4_000.0,
    db: bool = False,
) -> pd.DataFrame:
    """Integrate Welch power spectral density within named frequency bands."""
    return _spec_core.band_power(
        traces,
        bands,
        segment=segment,
        db=db,
    )


def welch_rolling(
    traces: nocte.core.traces.Traces,
    *,
    window: float = 10_000.0,
    step: float = 1_000.0,
    segment: float = 4_000.0,
    db: bool = False,
) -> nocte.core.traces.Traces:
    """
    Estimate Welch power spectral density through fixed sliding windows.

    The result contains one trace per ``(source trace, frequency)`` pair.
    Rolling-window centers form the result time coordinate. A rolling window
    touching missing source samples produces missing power values.
    """
    return _spec_core.welch_rolling(
        traces,
        window=window,
        step=step,
        segment=segment,
        db=db,
    )


def band_power_rolling(
    traces: nocte.core.traces.Traces,
    bands: Bands,
    *,
    window: float = 10_000.0,
    step: float = 1_000.0,
    segment: float = 4_000.0,
    db: bool = False,
) -> nocte.core.traces.Traces:
    """
    Integrate Welch power within named bands through fixed sliding windows.

    The result contains one trace per ``(source trace, band)`` pair.
    Rolling-window centers form the result time coordinate. The complete
    rolling frequency cube is never materialized.
    """
    return _spec_core.band_power_rolling(
        traces,
        bands,
        window=window,
        step=step,
        segment=segment,
        db=db,
    )
