"""Private numerical implementation for continuous cross-correlation."""

from __future__ import annotations

import dataclasses
import typing

import numba
import numpy as np
import pandas as pd

from nocte._coll.traces import Traces
from nocte._coll.windows import (
    Win,
)
from nocte._core import num
from nocte._core.matching import Matches
from nocte._core.sampling import SamplingRate, TimeGrid

CorrelationMethod = typing.Literal['pearson', 'dot']

PEARSON = 0
DOT = 1

EMPTY_KERNEL = np.empty(0, dtype=np.float64)
EMPTY_KERNEL.flags.writeable = False


class SignalCore(typing.NamedTuple):
    """One side of a matched correlation calculation."""

    values: np.ndarray
    positions: np.ndarray
    bounds: num.Bounds


class XCorrCore(typing.NamedTuple):
    """Nested numerical representation used by inline kernels."""

    left: SignalCore
    right: SignalCore
    offsets: np.ndarray


class PackedXCorrCore(typing.NamedTuple):
    """Flat transport representation for Numba parallel kernels."""

    left_values: np.ndarray
    right_values: np.ndarray

    left_positions: np.ndarray
    right_positions: np.ndarray

    left_first: np.ndarray
    left_stop: np.ndarray
    right_first: np.ndarray
    right_stop: np.ndarray

    offsets: np.ndarray


class Segment(typing.NamedTuple):
    """One pair of equally sized source segments to correlate."""

    left_row: int
    right_row: int

    left_start: int
    right_start: int

    n_samples: int


class MetricCore(typing.NamedTuple):
    """Numerical correlation definition."""

    method: int
    kernel: np.ndarray


class RollingCore(typing.NamedTuple):
    """Rolling geometry in source-sample coordinates."""

    window_start: int
    window_stop: int

    anchor_start: int
    anchor_step: int
    n_times: int


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedXCorr:
    """Validated correlation inputs plus their canonical lag grid."""

    core: XCorrCore
    lag_grid: TimeGrid


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedRolling:
    """Validated rolling geometry plus its output time grid."""

    core: RollingCore
    grid: TimeGrid


def pack_core(
    core: XCorrCore,
) -> PackedXCorrCore:
    """Flatten the numerical structure for parallel Numba transport."""
    return PackedXCorrCore(
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
def unpack_core(
    packed: PackedXCorrCore,
) -> XCorrCore:
    """Reconstruct the nested numerical view inside compiled code."""
    return XCorrCore(
        left=SignalCore(
            values=packed.left_values,
            positions=packed.left_positions,
            bounds=num.Bounds(
                first=packed.left_first,
                stop=packed.left_stop,
            ),
        ),
        right=SignalCore(
            values=packed.right_values,
            positions=packed.right_positions,
            bounds=num.Bounds(
                first=packed.right_first,
                stop=packed.right_stop,
            ),
        ),
        offsets=packed.offsets,
    )


@numba.njit(inline='always')
def matched_rows(
    core: XCorrCore,
    match: int,
) -> tuple[int, int]:
    """Return left and right source rows for one match."""
    return (
        int(core.left.positions[match]),
        int(core.right.positions[match]),
    )


@numba.njit(inline='always')
def full_segment(
    core: XCorrCore,
    match: int,
    lag: int,
) -> Segment:
    """Find finite overlap for one full-signal match and lag."""
    left_row, right_row = matched_rows(
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

    return Segment(
        left_row=left_row,
        right_row=right_row,
        left_start=left_start,
        right_start=left_start + offset,
        n_samples=left_stop - left_start,
    )


@numba.njit(inline='always')
def rolling_segment(
    core: XCorrCore,
    rolling: RollingCore,
    match: int,
    lag: int,
    time: int,
) -> Segment:
    """Locate one fixed-size rolling segment, or mark it invalid."""
    left_row, right_row = matched_rows(
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

    return Segment(
        left_row=left_row,
        right_row=right_row,
        left_start=left_start,
        right_start=right_start,
        n_samples=n_samples,
    )


@numba.njit(inline='always')
def dot_corr(
    core: XCorrCore,
    segment: Segment,
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
def pearson_corr(
    core: XCorrCore,
    segment: Segment,
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
def segment_corr(
    core: XCorrCore,
    segment: Segment,
    metric: MetricCore,
) -> float:
    """Dispatch one valid segment to its numerical correlation."""
    if segment.n_samples <= 0:
        return np.nan

    if metric.method == DOT:
        return dot_corr(
            core,
            segment,
            metric.kernel,
        )

    return pearson_corr(
        core,
        segment,
    )


@numba.njit(
    cache=True,
    parallel=True,
)
def cross_corr_nb(
    packed: PackedXCorrCore,
    metric: MetricCore,
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

        core = unpack_core(packed)
        segment = full_segment(
            core,
            match,
            lag,
        )

        result[
            match,
            lag,
        ] = segment_corr(
            core,
            segment,
            metric,
        )

    return result


@numba.njit(
    cache=True,
    parallel=True,
)
def cross_corr_rolling_nb(
    packed: PackedXCorrCore,
    rolling: RollingCore,
    metric: MetricCore,
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

        core = unpack_core(packed)
        segment = rolling_segment(
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
        ] = segment_corr(
            core,
            segment,
            metric,
        )

    return result


def method_code(
    method: CorrelationMethod,
) -> int:
    if method == 'pearson':
        return PEARSON

    if method == 'dot':
        return DOT

    raise ValueError("method must be 'pearson' or 'dot'")


def validate_sampling(
    left: Traces,
    right: Traces,
) -> tuple[
    SamplingRate,
    int,
]:
    """Validate common sampling and return the right-vs-left sample offset."""
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


def prepare(
    left: Traces,
    right: Traces,
    matches: Matches,
    lags: np.ndarray,
) -> PreparedXCorr:
    """Validate public objects and construct the numerical representation."""
    sampling, phase_offset = validate_sampling(
        left,
        right,
    )

    requested_lag_grid = TimeGrid.from_times(lags)
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

        bounds = num.Bounds.from_traces(
            left,
            participating,
            desc='source traces',
        )

        left_bounds = bounds
        right_bounds = bounds

    else:
        left_bounds = num.Bounds.from_traces(
            left,
            left_positions,
            desc='left traces',
        )
        right_bounds = num.Bounds.from_traces(
            right,
            right_positions,
            desc='right traces',
        )

    return PreparedXCorr(
        core=XCorrCore(
            left=SignalCore(
                values=left.values,
                positions=left_positions,
                bounds=left_bounds,
            ),
            right=SignalCore(
                values=right.values,
                positions=right_positions,
                bounds=right_bounds,
            ),
            offsets=offsets,
        ),
        lag_grid=lag_grid,
    )


def prepare_rolling(
    prepared: PreparedXCorr,
    left: Traces,
    right: Traces,
    *,
    window: Win,
    step: float,
) -> PreparedRolling:
    """Validate rolling geometry and construct its output sampling grid."""
    if not isinstance(
        window,
        Win,
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

    output_grid = TimeGrid(
        sampling=sampling.strided(anchor_step),
        start=left.start + sampling.samples_to_ms(anchor_start),
        n_samples=n_times,
    )

    return PreparedRolling(
        core=RollingCore(
            window_start=window_start,
            window_stop=window_stop,
            anchor_start=anchor_start,
            anchor_step=anchor_step,
            n_times=n_times,
        ),
        grid=output_grid,
    )


def prepare_metric(
    method: CorrelationMethod,
    *,
    kernel: np.ndarray | None = None,
    kernel_samples: int | None = None,
) -> MetricCore:
    """Validate the mathematical definition separately from geometry."""
    code = method_code(method)

    if kernel is None:
        return MetricCore(
            method=code,
            kernel=EMPTY_KERNEL,
        )

    if code != DOT:
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
            'kernel length must equal the number of samples in the rolling window: '
            f'expected {kernel_samples}, got {len(kernel)}'
        )

    if not np.isfinite(kernel).all():
        raise ValueError('kernel must contain only finite values')

    return MetricCore(
        method=code,
        kernel=np.ascontiguousarray(kernel),
    )


def prepare_rolling_xcorr(
    left: Traces,
    right: Traces,
    matches: Matches,
    *,
    lags: np.ndarray,
    window: Win,
    step: float,
    method: CorrelationMethod,
    kernel: np.ndarray | None,
) -> tuple[
    PreparedXCorr,
    PreparedRolling,
    MetricCore,
]:
    """Prepare the shared inputs required by rolling xcorr variants."""
    prepared = prepare(
        left,
        right,
        matches,
        lags,
    )
    rolling = prepare_rolling(
        prepared,
        left,
        right,
        window=window,
        step=step,
    )
    metric = prepare_metric(
        method,
        kernel=kernel,
        kernel_samples=rolling.core.window_stop - rolling.core.window_start,
    )

    return (
        prepared,
        rolling,
        metric,
    )


def match_meta(
    matches: Matches,
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


def cross_corr_meta(
    matches: Matches,
) -> pd.DataFrame:
    """Metadata for one full xcorr result per match."""
    meta = match_meta(matches)

    provenance_name = num.provenance_name(
        matches.name,
        reserved=(
            'xcorr',
            'lag',
        ),
    )

    num.set_provenance(
        meta,
        provenance_name,
        matches.index.to_numpy(copy=False),
    )

    meta.index = matches.index.rename('xcorr')

    return meta


def full_result(
    prepared: PreparedXCorr,
    metric: MetricCore,
    meta: pd.DataFrame,
) -> Traces:
    values = cross_corr_nb(
        pack_core(prepared.core),
        metric,
    )

    return Traces.from_grid(
        values,
        prepared.lag_grid,
        meta=meta,
    )


def rolling_result(
    prepared: PreparedXCorr,
    rolling: PreparedRolling,
    metric: MetricCore,
    *,
    source_meta: pd.DataFrame,
    source_ids: np.ndarray,
    source_name: str,
) -> Traces:
    values = cross_corr_rolling_nb(
        pack_core(prepared.core),
        rolling.core,
        metric,
    )

    return num.feature_traces(
        values,
        rolling.grid,
        source_meta=source_meta,
        source_ids=source_ids,
        source_name=source_name,
        feature_name='lag',
        features=prepared.lag_grid.times,
        result_name='xcorr',
    )


def cross_corr(
    left: Traces,
    right: Traces,
    matches: Matches,
    *,
    lags: np.ndarray,
    method: CorrelationMethod,
) -> Traces:
    prepared = prepare(
        left,
        right,
        matches,
        lags,
    )
    metric = prepare_metric(method)

    return full_result(
        prepared,
        metric,
        cross_corr_meta(matches),
    )


def auto_corr(
    traces: Traces,
    *,
    lags: np.ndarray,
    method: CorrelationMethod,
) -> Traces:
    matches = Matches.from_identity(traces)

    prepared = prepare(
        traces,
        traces,
        matches,
        lags,
    )
    metric = prepare_metric(method)

    return full_result(
        prepared,
        metric,
        traces.meta,
    )


def cross_corr_rolling(
    left: Traces,
    right: Traces,
    matches: Matches,
    *,
    lags: np.ndarray,
    window: Win,
    step: float,
    method: CorrelationMethod,
    kernel: np.ndarray | None,
) -> Traces:
    prepared, rolling, metric = prepare_rolling_xcorr(
        left,
        right,
        matches,
        lags=lags,
        window=window,
        step=step,
        method=method,
        kernel=kernel,
    )

    return rolling_result(
        prepared,
        rolling,
        metric,
        source_meta=match_meta(matches),
        source_ids=matches.index.to_numpy(copy=False),
        source_name=matches.name,
    )


def auto_corr_rolling(
    traces: Traces,
    *,
    lags: np.ndarray,
    window: Win,
    step: float,
    method: CorrelationMethod,
    kernel: np.ndarray | None,
) -> Traces:
    matches = Matches.from_identity(traces)

    prepared, rolling, metric = prepare_rolling_xcorr(
        traces,
        traces,
        matches,
        lags=lags,
        window=window,
        step=step,
        method=method,
        kernel=kernel,
    )

    return rolling_result(
        prepared,
        rolling,
        metric,
        source_meta=traces.meta,
        source_ids=traces.index.to_numpy(copy=False),
        source_name=traces.name,
    )
