"""Continuous full-signal and rolling cross-correlation for Traces.

Positive lag compares left(t) with right(t + lag), so a positive-lag peak means
that the left signal leads the right signal.
"""

from __future__ import annotations

import dataclasses
import typing

import numba
import numpy as np
import pandas as pd

import nocte.core.matching
import nocte.core.sampling
import nocte.core.traces
import nocte.core.windows

CorrelationMethod = typing.Literal['pearson', 'dot']

__all__ = [
    'auto_corr',
    'auto_corr_rolling',
    'cross_corr',
    'cross_corr_rolling',
]


_PEARSON = 0
_DOT = 1

_EMPTY_KERNEL = np.empty(0, dtype=np.float64)
_EMPTY_KERNEL.flags.writeable = False


# ----------------------------------------------------------------------
# numerical data structures


class _Bounds(typing.NamedTuple):
    """Finite half-open sample bounds for each source trace."""

    first: np.ndarray
    stop: np.ndarray


class _SignalCore(typing.NamedTuple):
    """One side of a matched correlation calculation."""

    values: np.ndarray
    positions: np.ndarray
    bounds: _Bounds


class _XCorrCore(typing.NamedTuple):
    """Nested numerical representation used by the inline kernels."""

    left: _SignalCore
    right: _SignalCore
    offsets: np.ndarray


class _PackedXCorrCore(typing.NamedTuple):
    """
    Flat transport representation for Numba parallel kernels.

    Numba handles nested NamedTuples well in normal njit code, but nested
    NamedTuple arguments currently fail during parallel prange lowering.
    The parallel drivers therefore receive this flat tuple and reconstruct
    the nested _XCorrCore locally.
    """

    left_values: np.ndarray
    right_values: np.ndarray

    left_positions: np.ndarray
    right_positions: np.ndarray

    left_first: np.ndarray
    left_stop: np.ndarray
    right_first: np.ndarray
    right_stop: np.ndarray

    offsets: np.ndarray


class _Segment(typing.NamedTuple):
    """One pair of equally sized source segments to correlate."""

    left_row: int
    right_row: int

    left_start: int
    right_start: int

    n_samples: int


class _MetricCore(typing.NamedTuple):
    """Numerical correlation definition."""

    method: int
    kernel: np.ndarray


class _RollingCore(typing.NamedTuple):
    """Rolling geometry in source-sample coordinates."""

    window_start: int
    window_stop: int

    anchor_start: int
    anchor_step: int
    n_times: int


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedXCorr:
    """Validated correlation inputs plus their canonical lag grid."""

    core: _XCorrCore
    lag_grid: nocte.core.traces.TimeGrid


@dataclasses.dataclass(frozen=True, slots=True)
class _PreparedRolling:
    """Validated rolling geometry plus its output time grid."""

    core: _RollingCore
    grid: nocte.core.traces.TimeGrid


# ----------------------------------------------------------------------
# numerical structure conversion


def _pack_core(
    core: _XCorrCore,
) -> _PackedXCorrCore:
    """
    Flatten the conceptual numerical structure for parallel Numba transport.

    Arrays are not copied.
    """
    return _PackedXCorrCore(
        left_values=core.left.values,
        right_values=core.right.values,
        left_positions=core.left.positions,
        right_positions=core.right.positions,
        left_first=core.left.bounds.first,
        left_stop=core.left.bounds.stop,
        right_first=core.right.bounds.first,
        right_stop=core.right.bounds.stop,
        offsets=core.offsets,
    )


@numba.njit(inline='always')
def _unpack_core(
    packed: _PackedXCorrCore,
) -> _XCorrCore:
    """Reconstruct the nested numerical view inside compiled code."""
    return _XCorrCore(
        left=_SignalCore(
            values=packed.left_values,
            positions=packed.left_positions,
            bounds=_Bounds(
                first=packed.left_first,
                stop=packed.left_stop,
            ),
        ),
        right=_SignalCore(
            values=packed.right_values,
            positions=packed.right_positions,
            bounds=_Bounds(
                first=packed.right_first,
                stop=packed.right_stop,
            ),
        ),
        offsets=packed.offsets,
    )


# ----------------------------------------------------------------------
# numerical geometry


@numba.njit(inline='always')
def _matched_rows(
    core: _XCorrCore,
    match: int,
) -> tuple[int, int]:
    """Return left and right source rows for one match."""
    return (
        int(core.left.positions[match]),
        int(core.right.positions[match]),
    )


@numba.njit(inline='always')
def _full_segment(
    core: _XCorrCore,
    match: int,
    lag: int,
) -> _Segment:
    """Find the finite overlap for one full-signal match and lag."""
    left_row, right_row = _matched_rows(
        core,
        match,
    )
    offset = int(core.offsets[lag])

    left_first = int(core.left.bounds.first[left_row])
    left_stop = int(core.left.bounds.stop[left_row])

    right_first = int(core.right.bounds.first[right_row])
    right_stop = int(core.right.bounds.stop[right_row])

    left_start = max(
        left_first,
        right_first - offset,
    )
    left_stop = min(
        left_stop,
        right_stop - offset,
    )

    return _Segment(
        left_row=left_row,
        right_row=right_row,
        left_start=left_start,
        right_start=left_start + offset,
        n_samples=left_stop - left_start,
    )


@numba.njit(inline='always')
def _rolling_segment(
    core: _XCorrCore,
    rolling: _RollingCore,
    match: int,
    lag: int,
    time: int,
) -> _Segment:
    """Locate one fixed-size rolling segment, or mark it invalid."""
    left_row, right_row = _matched_rows(
        core,
        match,
    )
    offset = int(core.offsets[lag])

    anchor = rolling.anchor_start + time * rolling.anchor_step

    left_start = anchor + rolling.window_start
    left_stop = anchor + rolling.window_stop

    right_start = left_start + offset
    right_stop = left_stop + offset

    n_samples = left_stop - left_start

    if left_start < int(core.left.bounds.first[left_row]) or left_stop > int(
        core.left.bounds.stop[left_row]
    ):
        n_samples = 0

    if right_start < int(core.right.bounds.first[right_row]) or right_stop > int(
        core.right.bounds.stop[right_row]
    ):
        n_samples = 0

    return _Segment(
        left_row=left_row,
        right_row=right_row,
        left_start=left_start,
        right_start=right_start,
        n_samples=n_samples,
    )


# ----------------------------------------------------------------------
# numerical correlation


@numba.njit(inline='always')
def _dot_corr(
    core: _XCorrCore,
    segment: _Segment,
    kernel: np.ndarray,
) -> float:
    """Raw or sample-weighted dot product over one segment."""
    value = 0.0

    if len(kernel) == 0:
        for i in range(segment.n_samples):
            value += (
                core.left.values[
                    segment.left_row,
                    segment.left_start + i,
                ]
                * core.right.values[
                    segment.right_row,
                    segment.right_start + i,
                ]
            )

        return value

    for i in range(segment.n_samples):
        value += (
            core.left.values[
                segment.left_row,
                segment.left_start + i,
            ]
            * core.right.values[
                segment.right_row,
                segment.right_start + i,
            ]
            * kernel[i]
        )

    return value


@numba.njit(inline='always')
def _pearson_corr(
    core: _XCorrCore,
    segment: _Segment,
) -> float:
    """Pearson correlation over one finite segment."""
    n_samples = segment.n_samples

    if n_samples < 2:
        return np.nan

    left_mean = 0.0
    right_mean = 0.0

    for i in range(n_samples):
        left_mean += core.left.values[
            segment.left_row,
            segment.left_start + i,
        ]
        right_mean += core.right.values[
            segment.right_row,
            segment.right_start + i,
        ]

    left_mean /= n_samples
    right_mean /= n_samples

    left_var = 0.0
    right_var = 0.0
    covariance = 0.0

    for i in range(n_samples):
        left = (
            core.left.values[
                segment.left_row,
                segment.left_start + i,
            ]
            - left_mean
        )
        right = (
            core.right.values[
                segment.right_row,
                segment.right_start + i,
            ]
            - right_mean
        )

        left_var += left * left
        right_var += right * right
        covariance += left * right

    if left_var <= 0.0 or right_var <= 0.0:
        return np.nan

    return covariance / np.sqrt(left_var * right_var)


@numba.njit(inline='always')
def _segment_corr(
    core: _XCorrCore,
    segment: _Segment,
    metric: _MetricCore,
) -> float:
    """Dispatch one valid segment to its numerical correlation."""
    if segment.n_samples <= 0:
        return np.nan

    if metric.method == _DOT:
        return _dot_corr(
            core,
            segment,
            metric.kernel,
        )

    return _pearson_corr(
        core,
        segment,
    )


# ----------------------------------------------------------------------
# parallel numerical drivers


@numba.njit(
    cache=True,
    parallel=True,
)
def _cross_corr_nb(
    packed: _PackedXCorrCore,
    metric: _MetricCore,
) -> np.ndarray:
    """Calculate full-signal correlation for all matches and lags."""
    n_matches = len(packed.left_positions)
    n_lags = len(packed.offsets)

    result = np.empty(
        (
            n_matches,
            n_lags,
        ),
        dtype=np.float64,
    )

    for flat in numba.prange(n_matches * n_lags):
        match = flat // n_lags
        lag = flat % n_lags

        core = _unpack_core(packed)
        segment = _full_segment(
            core,
            match,
            lag,
        )

        result[
            match,
            lag,
        ] = _segment_corr(
            core,
            segment,
            metric,
        )

    return result


@numba.njit(
    cache=True,
    parallel=True,
)
def _cross_corr_rolling_nb(
    packed: _PackedXCorrCore,
    rolling: _RollingCore,
    metric: _MetricCore,
) -> np.ndarray:
    """Calculate rolling correlation for all matches, lags, and times."""
    n_matches = len(packed.left_positions)
    n_lags = len(packed.offsets)
    n_times = rolling.n_times

    result = np.empty(
        (
            n_matches,
            n_lags,
            n_times,
        ),
        dtype=np.float64,
    )

    for flat in numba.prange(n_matches * n_lags * n_times):
        time = flat % n_times

        pair_lag = flat // n_times
        lag = pair_lag % n_lags
        match = pair_lag // n_lags

        core = _unpack_core(packed)
        segment = _rolling_segment(
            core,
            rolling,
            match,
            lag,
            time,
        )

        result[
            match,
            lag,
            time,
        ] = _segment_corr(
            core,
            segment,
            metric,
        )

    return result


# ----------------------------------------------------------------------
# public-object validation


def _method_code(
    method: CorrelationMethod,
) -> int:
    if method == 'pearson':
        return _PEARSON

    if method == 'dot':
        return _DOT

    raise ValueError("method must be 'pearson' or 'dot'")


def _validate_sampling(
    left: nocte.core.traces.Traces,
    right: nocte.core.traces.Traces,
) -> tuple[
    nocte.core.sampling.SamplingRate,
    int,
]:
    """
    Validate common sampling and return the right-vs-left sample offset.
    """
    if left.sampling != right.sampling:
        raise ValueError(
            'left and right must have identical sampling rates; '
            'resample explicitly before cross-correlation'
        )

    phase_offset = left.sampling.ms_to_samples_exact(
        left.start - right.start,
        desc='left/right temporal offset',
    )

    return (
        left.sampling,
        phase_offset,
    )


def _prepare(
    left: nocte.core.traces.Traces,
    right: nocte.core.traces.Traces,
    matches: nocte.core.matching.Matches,
    lags: np.ndarray,
) -> _PreparedXCorr:
    """Validate public objects and construct the numerical representation."""
    sampling, phase_offset = _validate_sampling(
        left,
        right,
    )

    requested_lag_grid = nocte.core.traces.TimeGrid.from_times(lags)

    lag_grid = requested_lag_grid.align_to(sampling)

    offsets = lag_grid.sample_offsets(sampling) + phase_offset

    left_positions = np.ascontiguousarray(
        matches.left_positions(left),
        dtype=np.intp,
    )
    right_positions = np.ascontiguousarray(
        matches.right_positions(right),
        dtype=np.intp,
    )

    if left is right:
        participating = np.concatenate(
            [
                left_positions,
                right_positions,
            ]
        )

        bounds = left.valid_bounds(
            participating,
            desc='source traces',
        )

        left_bounds = bounds
        right_bounds = bounds

    else:
        left_bounds = left.valid_bounds(
            left_positions,
            desc='left traces',
        )
        right_bounds = right.valid_bounds(
            right_positions,
            desc='right traces',
        )

    return _PreparedXCorr(
        core=_XCorrCore(
            left=_SignalCore(
                values=left.values,
                positions=left_positions,
                bounds=_Bounds(*left_bounds),
            ),
            right=_SignalCore(
                values=right.values,
                positions=right_positions,
                bounds=_Bounds(*right_bounds),
            ),
            offsets=offsets,
        ),
        lag_grid=lag_grid,
    )


def _prepare_rolling(
    prepared: _PreparedXCorr,
    left: nocte.core.traces.Traces,
    right: nocte.core.traces.Traces,
    *,
    window: nocte.core.windows.Win,
    step: float,
) -> _PreparedRolling:
    """Validate rolling geometry and construct its output sampling grid."""
    if not isinstance(
        window,
        nocte.core.windows.Win,
    ):
        raise TypeError('window must be a Win')

    if window.is_empty():
        raise ValueError('window must have positive duration')

    sampling = left.sampling

    window_start = sampling.ms_to_samples_exact(
        window.time_at('start'),
        desc='window start',
    )
    window_stop = sampling.ms_to_samples_exact(
        window.time_at('stop'),
        desc='window stop',
    )

    if window_stop <= window_start:
        raise ValueError('window must contain at least one source sample')

    step = float(step)

    if not np.isfinite(step) or step <= 0:
        raise ValueError('step must be finite and positive')

    anchor_step = sampling.ms_to_samples_exact(
        step,
        desc='rolling step',
    )

    if anchor_step <= 0:
        raise ValueError('rolling step must span at least one source sample')

    offsets = prepared.core.offsets

    min_offset = int(offsets[0])
    max_offset = int(offsets[-1])

    # Output geometry depends only on structural sampling, the requested
    # window, and the requested lags. Edge NaNs affect values but never
    # alter the shape or time grid of the result.
    anchor_start = max(
        -window_start,
        -window_start - min_offset,
    )

    anchor_last = min(
        left.n_samples - window_stop,
        right.n_samples - window_stop - max_offset,
    )

    if anchor_last < anchor_start:
        raise ValueError('window and lags leave no valid rolling positions')

    n_times = (anchor_last - anchor_start) // anchor_step + 1

    output_grid = nocte.core.traces.TimeGrid(
        sampling=(nocte.core.sampling.SamplingRate(sampling.rate / anchor_step)),
        start=(left.start + sampling.samples_to_ms(anchor_start)),
        n_samples=n_times,
    )

    return _PreparedRolling(
        core=_RollingCore(
            window_start=window_start,
            window_stop=window_stop,
            anchor_start=anchor_start,
            anchor_step=anchor_step,
            n_times=n_times,
        ),
        grid=output_grid,
    )


def _prepare_metric(
    method: CorrelationMethod,
    *,
    kernel: np.ndarray | None = None,
    kernel_samples: int | None = None,
) -> _MetricCore:
    """Validate the mathematical definition separately from geometry."""
    code = _method_code(method)

    if kernel is None:
        return _MetricCore(
            method=code,
            kernel=_EMPTY_KERNEL,
        )

    if code != _DOT:
        raise ValueError("kernel is only supported with method='dot'")

    if kernel_samples is None:
        raise ValueError('kernel requires a fixed-size correlation window')

    kernel = np.asarray(
        kernel,
        dtype=float,
    )

    if kernel.ndim != 1:
        raise ValueError('kernel must be one-dimensional')

    if len(kernel) != kernel_samples:
        raise ValueError(
            'kernel length must equal '
            'the number of samples in '
            'the rolling window: '
            f'expected {kernel_samples}, '
            f'got {len(kernel)}'
        )

    if not np.isfinite(kernel).all():
        raise ValueError('kernel must contain only finite values')

    return _MetricCore(
        method=code,
        kernel=np.ascontiguousarray(kernel),
    )


def _prepare_rolling_xcorr(
    left: nocte.core.traces.Traces,
    right: nocte.core.traces.Traces,
    matches: nocte.core.matching.Matches,
    *,
    lags: np.ndarray,
    window: nocte.core.windows.Win,
    step: float,
    method: CorrelationMethod,
    kernel: np.ndarray | None,
) -> tuple[
    _PreparedXCorr,
    _PreparedRolling,
    _MetricCore,
]:
    """Prepare the shared inputs required by rolling xcorr variants."""
    prepared = _prepare(
        left,
        right,
        matches,
        lags,
    )

    rolling = _prepare_rolling(
        prepared,
        left,
        right,
        window=window,
        step=step,
    )

    metric = _prepare_metric(
        method,
        kernel=kernel,
        kernel_samples=(rolling.core.window_stop - rolling.core.window_start),
    )

    return (
        prepared,
        rolling,
        metric,
    )


# ----------------------------------------------------------------------
# result metadata


def _provenance_name(
    name: str,
) -> str:
    """Avoid collisions with structural xcorr metadata."""
    if name in (
        'xcorr',
        'lag',
    ):
        return f'source_{name}'

    return name


def _set_provenance(
    meta: pd.DataFrame,
    name: str,
    values: np.ndarray,
) -> None:
    """Set provenance without silently contradicting existing metadata."""
    if name not in meta.columns:
        meta[name] = values
        return

    if not np.array_equal(
        meta[name].to_numpy(copy=False),
        values,
    ):
        raise ValueError(f'metadata column {name!r} contradicts xcorr provenance')


def _match_meta(
    matches: nocte.core.matching.Matches,
) -> pd.DataFrame:
    """Source identities plus pair-specific metadata, indexed by match."""
    return pd.concat(
        [
            matches.to_frame(),
            matches.meta,
        ],
        axis=1,
        verify_integrity=True,
    )


def _cross_corr_meta(
    matches: nocte.core.matching.Matches,
) -> pd.DataFrame:
    """Metadata for one full xcorr result per match."""
    meta = _match_meta(matches)

    _set_provenance(
        meta,
        _provenance_name(matches.name),
        matches.index.to_numpy(copy=False),
    )

    meta.index = matches.index.rename('xcorr')

    return meta


def _rolling_meta(
    meta: pd.DataFrame,
    *,
    source_ids: np.ndarray,
    source_name: str,
    lag_grid: nocte.core.traces.TimeGrid,
) -> pd.DataFrame:
    """Expand one source item into one derived trace per lag."""
    n_sources = len(meta)
    n_lags = lag_grid.n_samples

    take = np.repeat(
        np.arange(
            n_sources,
            dtype=np.intp,
        ),
        n_lags,
    )

    result = meta.iloc[take].copy()

    _set_provenance(
        result,
        _provenance_name(source_name),
        np.repeat(
            source_ids,
            n_lags,
        ),
    )

    if 'lag' in result.columns:
        raise ValueError("metadata column 'lag' conflicts with xcorr provenance")

    result['lag'] = np.tile(
        lag_grid.times,
        n_sources,
    )

    result.index = pd.RangeIndex(
        len(result),
        name='xcorr',
    )

    return result


# ----------------------------------------------------------------------
# result construction


def _from_grid(
    values: np.ndarray,
    grid: nocte.core.traces.TimeGrid,
    meta: pd.DataFrame,
) -> nocte.core.traces.Traces:
    return nocte.core.traces.Traces.from_array(
        values,
        hz=grid.sampling.rate,
        start=grid.start,
        meta=meta,
    )


def _full_result(
    prepared: _PreparedXCorr,
    metric: _MetricCore,
    meta: pd.DataFrame,
) -> nocte.core.traces.Traces:
    values = _cross_corr_nb(
        _pack_core(prepared.core),
        metric,
    )

    return _from_grid(
        values,
        prepared.lag_grid,
        meta,
    )


def _rolling_result(
    prepared: _PreparedXCorr,
    rolling: _PreparedRolling,
    metric: _MetricCore,
    meta: pd.DataFrame,
) -> nocte.core.traces.Traces:
    values = _cross_corr_rolling_nb(
        _pack_core(prepared.core),
        rolling.core,
        metric,
    )

    values = values.reshape(
        (
            values.shape[0] * values.shape[1],
            values.shape[2],
        )
    )

    return _from_grid(
        values,
        rolling.grid,
        meta,
    )


# ----------------------------------------------------------------------
# public API


def cross_corr(
    left: nocte.core.traces.Traces,
    right: nocte.core.traces.Traces,
    matches: nocte.core.matching.Matches,
    *,
    lags: np.ndarray,
    method: CorrelationMethod = 'pearson',
) -> nocte.core.traces.Traces:
    """
    Cross-correlate matched traces over their available finite overlap.

    Positive lag compares `left(t)` with `right(t + lag)`. Lags are expressed
    in milliseconds and must form a regular grid aligned to integer source
    samples.

    Leading and trailing NaNs reduce the available overlap. Internal NaN gaps
    are rejected.

    The result contains one trace per match, with lag as its sampled temporal
    coordinate.
    """
    prepared = _prepare(
        left,
        right,
        matches,
        lags,
    )
    metric = _prepare_metric(method)

    return _full_result(
        prepared,
        metric,
        _cross_corr_meta(matches),
    )


def auto_corr(
    traces: nocte.core.traces.Traces,
    *,
    lags: np.ndarray,
    method: CorrelationMethod = 'pearson',
) -> nocte.core.traces.Traces:
    """
    Autocorrelate every trace over its available finite overlap.

    This is the identity-match specialization of `cross_corr`. Trace identities
    and metadata are preserved.
    """
    matches = nocte.core.matching.Matches.from_identity(traces)

    prepared = _prepare(
        traces,
        traces,
        matches,
        lags,
    )
    metric = _prepare_metric(method)

    return _full_result(
        prepared,
        metric,
        traces.meta,
    )


def cross_corr_rolling(
    left: nocte.core.traces.Traces,
    right: nocte.core.traces.Traces,
    matches: nocte.core.matching.Matches,
    *,
    lags: np.ndarray,
    window: nocte.core.windows.Win,
    step: float,
    method: CorrelationMethod = 'pearson',
    kernel: np.ndarray | None = None,
) -> nocte.core.traces.Traces:
    """
    Cross-correlate matched traces through fixed windows in time.

    For output time `t` and lag `d`, the calculation compares `t + window`
    on the left with `t + window + d` on the right.

    The result contains one trace per `(match, lag)` pair. Its sampled temporal
    coordinate is rolling time and `lag` is stored as item metadata.

    Rolling windows are never shortened to accommodate leading or trailing
    NaNs. If either requested window is not fully observed, that result is NaN.

    `kernel` applies unnormalized sample-wise weights to `method='dot'` and
    must contain exactly one value per sample in `window`.
    """
    prepared, rolling, metric = _prepare_rolling_xcorr(
        left,
        right,
        matches,
        lags=lags,
        window=window,
        step=step,
        method=method,
        kernel=kernel,
    )

    meta = _rolling_meta(
        _match_meta(matches),
        source_ids=(matches.index.to_numpy(copy=False)),
        source_name=(matches.name),
        lag_grid=(prepared.lag_grid),
    )

    return _rolling_result(
        prepared,
        rolling,
        metric,
        meta,
    )


def auto_corr_rolling(
    traces: nocte.core.traces.Traces,
    *,
    lags: np.ndarray,
    window: nocte.core.windows.Win,
    step: float,
    method: CorrelationMethod = 'pearson',
    kernel: np.ndarray | None = None,
) -> nocte.core.traces.Traces:
    """
    Rolling autocorrelation for every trace and requested lag.

    The result contains one trace per `(source trace, lag)` pair.
    """
    matches = nocte.core.matching.Matches.from_identity(traces)

    prepared, rolling, metric = _prepare_rolling_xcorr(
        traces,
        traces,
        matches,
        lags=lags,
        window=window,
        step=step,
        method=method,
        kernel=kernel,
    )

    meta = _rolling_meta(
        traces.meta,
        source_ids=(traces.index.to_numpy(copy=False)),
        source_name=(traces.name),
        lag_grid=(prepared.lag_grid),
    )

    return _rolling_result(
        prepared,
        rolling,
        metric,
        meta,
    )
