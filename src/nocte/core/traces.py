from __future__ import annotations

import collections.abc
import fractions
import itertools
import logging
import pathlib
import typing
import warnings

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal

import nocte.core.hdf
from nocte.core.collections import Collection
from nocte.core.sampling import SamplingRate
from nocte.core.windows import Win

logger = logging.getLogger(__name__)

FloatT = typing.TypeVar('FloatT', bound=np.floating[typing.Any])
FloatU = typing.TypeVar('FloatU', bound=np.floating[typing.Any])

InterpMethod = typing.Literal['linear', 'nearest']
TimeLike = float | collections.abc.Sequence[float] | np.ndarray | pd.Index | pd.Series


def _validate_trace_array(values: np.ndarray) -> np.ndarray:
    """Normalize public numerical input to a 2D floating-point array."""

    values = np.asarray(values)

    if values.ndim == 1:
        values = values.reshape(1, -1)

    elif values.ndim != 2:
        raise ValueError('trace values must be one- or two-dimensional')

    if np.issubdtype(values.dtype, np.complexfloating):
        raise TypeError('complex trace values are not supported')

    if np.issubdtype(values.dtype, np.bool_):
        warnings.warn(
            'Boolean trace values are converted to float64',
            UserWarning,
            stacklevel=3,
        )
        values = values.astype(np.float64)

    elif np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.float64)

    elif not np.issubdtype(values.dtype, np.floating):
        raise TypeError('trace values must be real numeric data')

    return np.ascontiguousarray(values)


def _validate_method(method: InterpMethod) -> None:
    if method not in ('linear', 'nearest'):
        raise ValueError("method must be 'linear' or 'nearest'")


def _validate_max_gap(max_gap: float) -> float:

    max_gap = float(max_gap)

    if max_gap > 0:
        raise ValueError('max_gap must be positive')

    if np.isinf(max_gap):
        return max_gap

    if not np.isfinite(max_gap):
        raise ValueError('max_gap must be finite or positive infinite')

    return max_gap


def _regular_times(start: float, last: float, sampling: SamplingRate) -> np.ndarray:
    """Return regular sample coordinates from `start` through `last`, inclusive."""
    if last < start:
        return np.empty(0, dtype=float)

    ratio = (last - start) / sampling.period_ms
    nearest = np.round(ratio)
    if np.isclose(ratio, nearest, rtol=1e-10, atol=1e-10):
        ratio = nearest

    n_samples = int(np.floor(ratio)) + 1
    return start + sampling.samples_to_ms(np.arange(n_samples, dtype=np.intp))


def _interpolate_irregular(
    values: np.ndarray,
    times: np.ndarray,
    target: np.ndarray,
    *,
    method: InterpMethod,
    max_gap: float = np.inf,
) -> np.ndarray:
    """Interpolate rows sharing one irregular source coordinate."""
    _validate_method(method)
    max_gap = _validate_max_gap(max_gap)

    values = np.asarray(values)
    times = np.asarray(times, dtype=float)
    target = np.asarray(target, dtype=float)

    result = np.full(
        (values.shape[0], len(target)),
        np.nan,
        dtype=values.dtype,
    )

    if len(times) == 0 or len(target) == 0:
        return result

    positions = np.searchsorted(times, target, side='left')
    clipped = np.clip(positions, 0, len(times) - 1)
    exact = (positions < len(times)) & np.isclose(
        times[clipped],
        target,
        rtol=1e-10,
        atol=1e-12,
    )

    if exact.any():
        result[:, exact] = values[:, positions[exact]]

    interp = ~exact & (positions > 0) & (positions < len(times))
    if interp.any():
        right = positions[interp]
        left = right - 1
        gaps = times[right] - times[left]

        allowed = np.ones(len(right), dtype=bool)
        allowed &= gaps <= max_gap + np.finfo(float).eps * 16

        if allowed.any():
            target_allowed = target[interp][allowed]
            left = left[allowed]
            right = right[allowed]
            out_cols = np.flatnonzero(interp)[allowed]

            if method == 'linear':
                fraction = (target_allowed - times[left]) / (times[right] - times[left])
                result[:, out_cols] = (
                    values[:, left] + (values[:, right] - values[:, left]) * fraction
                )
            else:
                left_dist = target_allowed - times[left]
                right_dist = times[right] - target_allowed
                source = np.where(left_dist <= right_dist, left, right)
                result[:, out_cols] = values[:, source]

    # Each trace has its own finite support when edge samples are NaN.
    valid = ~np.isnan(values)
    any_valid = valid.any(axis=1)
    first = np.full(values.shape[0], -1, dtype=np.intp)
    last = np.full(values.shape[0], -1, dtype=np.intp)

    if values.shape[1]:
        first[any_valid] = np.argmax(valid[any_valid], axis=1)
        last[any_valid] = (
            values.shape[1] - 1 - np.argmax(valid[any_valid, ::-1], axis=1)
        )

    for row in np.flatnonzero(any_valid):
        inside = (target >= times[first[row]]) & (target <= times[last[row]])
        result[row, ~inside] = np.nan

    return result


class _TracesData(typing.Generic[FloatT]):
    """
    Internal storage for regularly sampled traces.

    Values are a floating-point array with shape `(n_traces, n_samples)`.
    All traces share one sampling rate and temporal origin. NaN represents
    missing observations without changing the regular sampling grid.
    """

    def __init__(
        self,
        values: npt.NDArray[FloatT],
        sampling: SamplingRate,
        start: float,
    ):
        values = np.asarray(values)

        if values.ndim != 2:
            raise ValueError('trace values must be a two-dimensional array')
        if not np.issubdtype(values.dtype, np.floating):
            raise TypeError('internal trace values must have a floating dtype')
        if not isinstance(sampling, SamplingRate):
            raise TypeError('sampling must be a SamplingRate')

        start = float(start)
        if not np.isfinite(start):
            raise ValueError('trace start must be finite')

        values = np.array(values, copy=True, order='C', subok=False)
        values.flags.writeable = False

        self._values = values
        self._sampling = sampling
        self._start = start

    def __len__(self) -> int:
        return self._values.shape[0]

    @property
    def values(self) -> npt.NDArray[FloatT]:
        return self._values

    @property
    def sampling(self) -> SamplingRate:
        return self._sampling

    @property
    def start(self) -> float:
        return self._start

    @property
    def stop(self) -> float:
        return self.start + self.sampling.samples_to_ms(self.n_samples)

    @property
    def n_samples(self) -> int:
        return self._values.shape[1]

    @property
    def time(self) -> np.ndarray:
        return self.start + self.sampling.samples_to_ms(
            np.arange(self.n_samples, dtype=np.intp)
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._values.shape

    @property
    def dtype(self) -> np.dtype[FloatT]:
        return self._values.dtype

    def astype(
        self,
        dtype: type[FloatU] | np.dtype[FloatU],
    ) -> _TracesData[FloatU]:

        dtype = np.dtype(dtype)

        if not np.issubdtype(dtype, np.floating):
            raise TypeError('Traces can only be cast to floating dtypes')

        return _TracesData(
            self.values.astype(dtype),
            self.sampling,
            self.start,
        )

    def sel_pos(self, positions: np.ndarray) -> typing.Self:
        positions = np.asarray(positions, dtype=np.intp)
        return self.__class__(
            self._values[positions],
            self.sampling,
            self.start,
        )

    def get_pos(self, position: int) -> pd.Series:
        return pd.Series(
            self._values[position],
            index=pd.Index(self.time, name='time_ms'),
            name=position,
        )

    def copy(self) -> typing.Self:
        return self

    def _with_start(self, start: float) -> typing.Self:
        start = float(start)
        if not np.isfinite(start):
            raise ValueError('trace start must be finite')

        result = self.__class__.__new__(self.__class__)
        result._values = self._values
        result._sampling = self._sampling
        result._start = start
        return result

    def sel_time(self, win: Win) -> typing.Self:
        win_start = win.time_at('start')
        win_stop = win.time_at('stop')

        if self.n_samples == 0:
            return self.copy()

        period = self.sampling.period_ms
        rel_start = (win_start - self.start) / period
        rel_stop = (win_stop - self.start) / period

        for name, value in (('start', rel_start), ('stop', rel_stop)):
            nearest = np.round(value)
            if np.isclose(value, nearest, rtol=1e-10, atol=1e-10):
                if name == 'start':
                    rel_start = nearest
                else:
                    rel_stop = nearest

        first = int(np.ceil(rel_start))
        stop = int(np.ceil(rel_stop))
        first = int(np.clip(first, 0, self.n_samples))
        stop = int(np.clip(stop, first, self.n_samples))

        new_start = self.start + self.sampling.samples_to_ms(first)
        return self.__class__(
            self._values[:, first:stop],
            self.sampling,
            new_start,
        )

    def shift(self, by: float) -> typing.Self:
        by = float(by)
        if not np.isfinite(by):
            raise ValueError('shift must be finite')

        return self._with_start(self.start + by)

    def _lookup_common(
        self,
        target: np.ndarray,
        *,
        method: InterpMethod,
    ) -> np.ndarray:
        _validate_method(method)
        target = np.asarray(target, dtype=float)
        result = np.full(
            (len(self), len(target)),
            np.nan,
            dtype=self.dtype,
        )

        if self.n_samples == 0 or len(target) == 0:
            return result

        finite_target = np.isfinite(target)
        fractional = (target - self.start) / self.sampling.period_ms
        safe_fractional = np.where(finite_target, fractional, 0.0)
        nearest_integer = np.round(safe_fractional)
        exact = finite_target & np.isclose(
            fractional,
            nearest_integer,
            rtol=1e-10,
            atol=1e-10,
        )
        exact_indices = nearest_integer.astype(np.intp, copy=False)
        exact &= (exact_indices >= 0) & (exact_indices < self.n_samples)

        if exact.any():
            result[:, exact] = self._values[:, exact_indices[exact]]

        remaining = finite_target & ~exact
        if method == 'linear':
            left = np.floor(safe_fractional).astype(np.intp, copy=False)
            right = left + 1
            interp = remaining & (left >= 0) & (right < self.n_samples)

            if interp.any():
                fraction = fractional[interp] - left[interp]
                result[:, interp] = (
                    self._values[:, left[interp]]
                    + (self._values[:, right[interp]] - self._values[:, left[interp]])
                    * fraction
                )
        else:
            left = np.floor(safe_fractional).astype(np.intp, copy=False)
            fraction = safe_fractional - left
            nearest = left + (fraction > 0.5).astype(np.intp)
            interp = remaining & (nearest >= 0) & (nearest < self.n_samples)
            if interp.any():
                result[:, interp] = self._values[:, nearest[interp]]

        first = self.first_valid_index()
        last = self.last_valid_index()
        nonempty = first >= 0

        for row in np.flatnonzero(nonempty):
            first_time = self.start + self.sampling.samples_to_ms(int(first[row]))
            last_time = self.start + self.sampling.samples_to_ms(int(last[row]))
            inside = (target >= first_time) & (target <= last_time)
            result[row, ~inside] = np.nan

        # Warn only when NaNs are encountered inside a trace's valid support.
        warned = False
        for row in np.flatnonzero(nonempty):
            first_time = self.start + self.sampling.samples_to_ms(int(first[row]))
            last_time = self.start + self.sampling.samples_to_ms(int(last[row]))
            inside = finite_target & (target >= first_time) & (target <= last_time)
            if inside.any() and np.isnan(result[row, inside]).any():
                warned = True
                break

        if warned:
            logger.warning(
                'Interpolation encountered missing samples; affected values remain NaN. '
                'Use fill_missing() first to bridge internal gaps.'
            )

        return result

    def lookup(
        self,
        time: TimeLike,
        *,
        method: InterpMethod = 'linear',
    ) -> np.ndarray:
        target = np.asarray(time, dtype=float)
        scalar = target.ndim == 0

        if scalar:
            target = target.reshape(1)
        elif target.ndim != 1:
            raise ValueError('time must be scalar or one-dimensional')

        result = self._lookup_common(
            target,
            method=method,
        )
        return result[:, 0] if scalar else result

    def lookup_each(
        self,
        time: np.ndarray,
        *,
        method: InterpMethod = 'linear',
    ) -> np.ndarray:
        _validate_method(method)
        time = np.asarray(time, dtype=float)

        if time.ndim != 1 or len(time) != len(self):
            raise ValueError('time must contain one value per trace')

        result = np.full(len(self), np.nan, dtype=self.dtype)
        first = self.first_valid_index()
        last = self.last_valid_index()

        for row, target in enumerate(time):
            if not np.isfinite(target) or first[row] < 0:
                continue

            first_time = self.start + self.sampling.samples_to_ms(int(first[row]))
            last_time = self.start + self.sampling.samples_to_ms(int(last[row]))
            if target < first_time or target > last_time:
                continue

            fractional = (target - self.start) / self.sampling.period_ms
            nearest_integer = round(fractional)

            if np.isclose(fractional, nearest_integer, rtol=1e-10, atol=1e-10):
                if 0 <= nearest_integer < self.n_samples:
                    result[row] = self._values[row, nearest_integer]
                continue

            left = int(np.floor(fractional))
            right = left + 1

            if method == 'linear':
                if left < 0 or right >= self.n_samples:
                    continue
                fraction = fractional - left
                result[row] = (
                    self._values[row, left]
                    + (self._values[row, right] - self._values[row, left]) * fraction
                )
            else:
                nearest = left if fractional - left <= 0.5 else right
                if 0 <= nearest < self.n_samples:
                    result[row] = self._values[row, nearest]

        if np.isnan(result).any():
            internal_nan = False
            for row, target in enumerate(time):
                if first[row] < 0 or not np.isfinite(target):
                    continue
                first_time = self.start + self.sampling.samples_to_ms(int(first[row]))
                last_time = self.start + self.sampling.samples_to_ms(int(last[row]))
                if first_time <= target <= last_time and np.isnan(result[row]):
                    internal_nan = True
                    break
            if internal_nan:
                logger.warning(
                    'Interpolation encountered missing samples; affected values remain NaN. '
                    'Use fill_missing() first to bridge internal gaps.'
                )

        return result

    def resample(self, hz: float) -> typing.Self:
        target_sampling = SamplingRate(hz)

        if self.n_samples == 0:
            return self.__class__(
                np.empty((len(self), 0), dtype=self.dtype),
                target_sampling,
                self.start,
            )

        last_time = self.start + self.sampling.samples_to_ms(self.n_samples - 1)
        target_time = _regular_times(
            self.start,
            last_time,
            target_sampling,
        )

        if np.isclose(target_sampling.rate, self.sampling.rate):
            return self.copy()

        if target_sampling.rate > self.sampling.rate:
            values = self._lookup_common(target_time, method='linear')
            return self.__class__(values, target_sampling, self.start)

        gapless = self.is_gapless()
        if not gapless.all():
            bad = int(np.count_nonzero(~gapless))
            raise ValueError(
                f'Cannot anti-alias {bad} trace(s) with internal NaN gaps; '
                'use fill_missing() first or select traces without internal gaps'
            )

        ratio = fractions.Fraction(
            target_sampling.rate / self.sampling.rate
        ).limit_denominator(100_000)
        up = ratio.numerator
        down = ratio.denominator
        intermediate_hz = self.sampling.rate * up / down
        intermediate_sampling = SamplingRate(intermediate_hz)

        output = np.full(
            (len(self), len(target_time)),
            np.nan,
            dtype=self.dtype,
        )
        first = self.first_valid_index()
        last = self.last_valid_index()

        for row in range(len(self)):
            if first[row] < 0:
                continue

            i0 = int(first[row])
            i1 = int(last[row])
            source = self._values[row, i0 : i1 + 1]
            source_start = self.start + self.sampling.samples_to_ms(i0)
            source_last = self.start + self.sampling.samples_to_ms(i1)

            if len(source) < 2:
                exact = np.isclose(
                    target_time,
                    source_start,
                    rtol=1e-10,
                    atol=1e-12,
                )
                output[row, exact] = source[0]
                continue

            sampled = scipy.signal.resample_poly(
                source,
                up,
                down,
            )
            sampled = np.asarray(sampled, dtype=self.dtype)
            sampled_time = source_start + intermediate_sampling.samples_to_ms(
                np.arange(len(sampled), dtype=np.intp)
            )
            inside = sampled_time <= source_last + np.finfo(float).eps * 16
            sampled = sampled[inside]
            sampled_time = sampled_time[inside]

            output[row] = _interpolate_irregular(
                sampled.reshape(1, -1),
                sampled_time,
                target_time,
                method='linear',
            )[0]

        return self.__class__(
            output,
            target_sampling,
            self.start,
        )

    def fill_missing(
        self,
        method: InterpMethod = 'linear',
        *,
        max_gap: float = np.inf,
    ) -> typing.Self:
        """
        Fill internal NaN gaps in each trace using interpolation.

        param method: Interpolation method, either 'linear' or 'nearest'.
        param max_gap: Maximum gap size in milliseconds to fill; larger gaps remain NaN.
        """
        _validate_method(method)
        max_gap = _validate_max_gap(max_gap)

        values = self._values.copy()
        period = self.sampling.period_ms

        for row in range(len(self)):
            valid = np.flatnonzero(~np.isnan(values[row]))
            if len(valid) < 2:
                continue

            for left, right in itertools.pairwise(valid):
                if right == left + 1:
                    continue

                gap = (right - left) * period
                if gap > max_gap + np.finfo(float).eps * 16:
                    continue

                positions = np.arange(left + 1, right, dtype=np.intp)
                if method == 'linear':
                    fraction = (positions - left) / (right - left)
                    values[row, positions] = (
                        values[row, left]
                        + (values[row, right] - values[row, left]) * fraction
                    )
                else:
                    use_left = positions - left <= right - positions
                    values[row, positions] = np.where(
                        use_left,
                        values[row, left],
                        values[row, right],
                    )

        return self.__class__(
            values,
            self.sampling,
            self.start,
        )

    def first_valid_index(self) -> np.ndarray:
        result = np.full(len(self), -1, dtype=np.intp)
        if self.n_samples == 0 or len(self) == 0:
            return result

        valid = ~np.isnan(self._values)
        nonempty = valid.any(axis=1)
        result[nonempty] = np.argmax(valid[nonempty], axis=1)
        return result

    def last_valid_index(self) -> np.ndarray:
        result = np.full(len(self), -1, dtype=np.intp)
        if self.n_samples == 0 or len(self) == 0:
            return result

        valid = ~np.isnan(self._values)
        nonempty = valid.any(axis=1)
        result[nonempty] = (
            self.n_samples
            - 1
            - np.argmax(
                valid[nonempty, ::-1],
                axis=1,
            )
        )
        return result

    def is_gapless(self) -> np.ndarray:
        first = self.first_valid_index()
        last = self.last_valid_index()
        result = np.ones(len(self), dtype=bool)

        for row in np.flatnonzero(first >= 0):
            result[row] = not np.isnan(
                self._values[row, first[row] : last[row] + 1]
            ).any()

        return result

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        """
        Store the trace payload.

        The payload is a two-dimensional HDF5 dataset containing the trace
        values. Shared sampling state is stored as attributes on that dataset.

        The target key must not already exist.
        """
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='a') as file:
            if key in file:
                raise FileExistsError(f'HDF5 key {key!r} already exists')

            dataset = file.create_dataset(
                key,
                data=self._values,
            )

            dataset.attrs['sampling_rate_hz'] = self.sampling.rate
            dataset.attrs['start_ms'] = self.start

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> typing.Self:
        """
        Load a trace payload previously stored with to_hdf().
        """
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='r') as file:
            if key not in file:
                raise KeyError(f'HDF5 key {key!r} does not exist')

            node = file[key]

            if not isinstance(node, h5py.Dataset):
                raise TypeError(f'Traces payload {key!r} must be an HDF5 dataset')

            sampling_rate_raw = node.attrs.get('sampling_rate_hz')

            if sampling_rate_raw is None:
                raise ValueError(f'Traces payload {key!r} is missing sampling_rate_hz')

            start_raw = node.attrs.get('start_ms')

            if start_raw is None:
                raise ValueError(f'Traces payload {key!r} is missing start_ms')

            values = np.asarray(node[...])

        if values.ndim != 2:
            raise ValueError(f'Traces payload {key!r} must be two-dimensional')

        if not np.issubdtype(
            values.dtype,
            np.floating,
        ):
            raise TypeError(f'Traces payload {key!r} must have a floating dtype')

        sampling_rate = float(sampling_rate_raw)
        start = float(start_raw)

        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError('stored sampling_rate_hz must be finite and positive')

        if not np.isfinite(start):
            raise ValueError('stored start_ms must be finite')

        return cls(
            values,
            SamplingRate(sampling_rate),
            start,
        )


class Traces(Collection[pd.Series], typing.Generic[FloatT]):
    """
    Indexed collection of regularly sampled continuous signals.

    Each item is one trace and `meta.iloc[i]` describes payload row `i`.
    The payload is a 2D floating-point NumPy array with one shared regular
    sampling grid; NaN represents missing observations. Floating input dtypes
    are preserved, integer inputs become float64, boolean inputs warn and
    become float64, and complex or non-numeric inputs are rejected.

    Traces is a data container with explicit sampling semantics. It does not
    emulate a NumPy array or pandas DataFrame.
    """

    def __init__(
        self,
        data: _TracesData[FloatT],
        meta: pd.DataFrame,
    ):
        if not isinstance(data, _TracesData):
            raise TypeError('data must be _TracesData')

        self._data = data
        self.meta = meta.copy()
        self._validate_meta(len(self._data))

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def from_array(
        cls,
        values: np.ndarray,
        hz: float,
        *,
        start: float = 0.0,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        """Build traces from samples already on one regular sampling grid."""
        values = _validate_trace_array(values)
        data = _TracesData(
            values,
            SamplingRate(hz),
            start,
        )

        return cls(
            data,
            cls._default_meta(len(data)) if meta is None else meta,
        )

    @classmethod
    def from_irregular(
        cls,
        values: np.ndarray,
        times: np.ndarray,
        hz: float,
        *,
        method: InterpMethod = 'linear',
        max_gap: float = np.inf,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        """Interpolate shared irregular observations onto one regular grid."""
        values = _validate_trace_array(values)
        times = np.asarray(times, dtype=float)
        sampling = SamplingRate(hz)
        _validate_method(method)
        max_gap = _validate_max_gap(max_gap)

        if times.ndim != 1:
            raise ValueError('times must be one-dimensional')
        if len(times) != values.shape[1]:
            raise ValueError('times length must match the number of samples')
        if len(times) == 0:
            raise ValueError('times cannot be empty')
        if not np.isfinite(times).all():
            raise ValueError('times must be finite')
        if np.any(np.diff(times) <= 0):
            raise ValueError('times must be strictly increasing')

        target = _regular_times(
            float(times[0]),
            float(times[-1]),
            sampling,
        )
        interpolated = _interpolate_irregular(
            values,
            times,
            target,
            method=method,
            max_gap=max_gap,
        )
        data = _TracesData(
            interpolated,
            sampling,
            float(times[0]),
        )

        return cls(
            data,
            cls._default_meta(len(data)) if meta is None else meta,
        )

    # ------------------------------------------------------------------
    # collection / representation

    def _sel_pos(self, positions: np.ndarray) -> typing.Self:
        return self.__class__(
            self._data.sel_pos(positions),
            self.meta.iloc[positions],
        )

    def copy(self) -> typing.Self:
        return self.__class__(
            self._data.copy(),
            self.meta.copy(),
        )

    def _get_pos(self, position: int) -> pd.Series:
        """Return one trace as a time-indexed Series."""
        trace = self._data.get_pos(position)

        trace.name = self.meta.index[position]

        return trace

    def to_frame(self) -> pd.DataFrame:
        """Return samples as a time-by-trace DataFrame."""
        return pd.DataFrame(
            self.values.T,
            index=pd.Index(self.time, name='time_ms'),
            columns=self.index.copy(),
            copy=False,
        )

    # ------------------------------------------------------------------
    # sampled state

    @property
    def values(self) -> npt.NDArray[FloatT]:
        return self._data.values

    @property
    def sampling(self) -> SamplingRate:
        return self._data.sampling

    @property
    def hz(self) -> float:
        return self.sampling.rate

    @property
    def period_ms(self) -> float:
        return self.sampling.period_ms

    @property
    def start(self) -> float:
        return self._data.start

    @property
    def stop(self) -> float:
        return self._data.stop

    @property
    def duration(self) -> float:
        return self.stop - self.start

    @property
    def extent(self) -> Win:
        return Win(
            self.start,
            self.stop,
        )

    @property
    def n_samples(self) -> int:
        return self._data.n_samples

    @property
    def time(self) -> np.ndarray:
        return self._data.time

    @property
    def shape(self) -> tuple[int, int]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype[FloatT]:
        return self._data.dtype

    def astype(
        self,
        dtype: type[FloatU] | np.dtype[FloatU],
    ) -> Traces[FloatU]:
        return Traces[FloatU](
            data=self._data.astype(dtype),
            meta=self.meta,
        )

    # ------------------------------------------------------------------
    # temporal selection / sampling

    def sel_time(self, win: Win) -> typing.Self:
        """Select existing samples contained in a window without interpolation."""
        return self.__class__(
            self._data.sel_time(win),
            self.meta,
        )

    def shift(self, by: float) -> typing.Self:
        """Shift the temporal coordinate without changing sample values."""
        return self.__class__(
            self._data.shift(by),
            self.meta,
        )

    def lookup(
        self,
        time: TimeLike,
        *,
        method: InterpMethod = 'linear',
    ) -> pd.Series | pd.DataFrame:
        """Interpolate all traces at one or more arbitrary time points."""
        requested = np.asarray(time, dtype=float)
        scalar = requested.ndim == 0
        values = self._data.lookup(
            time,
            method=method,
        )

        if scalar:
            return pd.Series(
                values,
                index=self.index.copy(),
                name=float(requested),
            )

        if requested.ndim != 1:
            raise ValueError('time must be scalar or one-dimensional')

        return pd.DataFrame(
            values.T,
            index=pd.Index(requested, name='time_ms'),
            columns=self.index.copy(),
        )

    def lookup_each(
        self,
        time: pd.Series,
        *,
        method: InterpMethod = 'linear',
    ) -> pd.Series:
        """Interpolate each trace at its corresponding time coordinate."""
        if not isinstance(time, pd.Series):
            raise TypeError('time must be a pandas Series indexed by trace ID')
        time = self._align_series(time, 'time')

        return pd.Series(
            self._data.lookup_each(
                time.to_numpy(dtype=float),
                method=method,
            ),
            index=self.index.copy(),
            name='value',
        )

    def resample(self, hz: float) -> typing.Self:
        """Represent the signals on a new regular sampling rate."""
        return self.__class__(
            self._data.resample(hz),
            self.meta,
        )

    def fill_missing(
        self,
        method: InterpMethod = 'linear',
        *,
        max_gap: float = np.inf,
    ) -> typing.Self:
        """Interpolate bounded internal NaN gaps without extrapolating edges."""
        return self.__class__(
            self._data.fill_missing(
                method,
                max_gap=max_gap,
            ),
            self.meta,
        )

    # ------------------------------------------------------------------
    # missing-data state

    def is_empty(self) -> pd.Series:
        is_empty = np.asarray(
            np.isnan(self.values).all(axis=1),
            dtype=bool,
        )

        return pd.Series(
            is_empty,
            index=self.index.copy(),
            name='is_empty',
        )

    def are_empty(self) -> bool:
        return bool(self.is_empty().all())

    def is_complete(self) -> pd.Series:
        is_partial = np.asarray(
            np.isnan(self.values).any(axis=1),
            dtype=bool,
        )

        is_complete = ~is_partial

        return pd.Series(
            is_complete,
            index=self.index.copy(),
            name='is_complete',
        )

    def are_complete(self) -> bool:
        return bool(self.is_complete().all())

    def is_gapless(self) -> pd.Series:
        return pd.Series(
            self._data.is_gapless(),
            index=self.index.copy(),
            name='is_gapless',
        )

    def are_gapless(self) -> bool:
        return bool(self.is_gapless().all())

    def first_valid_time(self) -> pd.Series:
        indices = self._data.first_valid_index()
        result = np.full(len(self), np.nan, dtype=float)
        valid = indices >= 0
        result[valid] = self.start + self.sampling.samples_to_ms(indices[valid])

        return pd.Series(
            result,
            index=self.index.copy(),
            name='first_valid_time',
        )

    def last_valid_time(self) -> pd.Series:
        indices = self._data.last_valid_index()
        result = np.full(len(self), np.nan, dtype=float)
        valid = indices >= 0
        result[valid] = self.start + self.sampling.samples_to_ms(indices[valid])

        return pd.Series(
            result,
            index=self.index.copy(),
            name='last_valid_time',
        )

    # ------------------------------------------------------------------
    # per-trace summaries

    def _summary(
        self,
        values: np.ndarray,
        *,
        name: str,
    ) -> pd.Series:
        return pd.Series(
            values,
            index=self.index.copy(),
            name=name,
        )

    def mean(self) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.nanmean(self.values, axis=1)
        return self._summary(values, name='mean')

    def median(self) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.nanmedian(self.values, axis=1)
        return self._summary(values, name='median')

    def std(self) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.nanstd(self.values, axis=1, ddof=1)
        return self._summary(values, name='std')

    def var(self) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.nanvar(self.values, axis=1, ddof=1)
        return self._summary(values, name='var')

    def min(self) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.nanmin(self.values, axis=1)
        return self._summary(values, name='min')

    def max(self) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.nanmax(self.values, axis=1)
        return self._summary(values, name='max')

    def quantile(self, q: float) -> pd.Series:
        q = float(q)
        if not 0 <= q <= 1:
            raise ValueError('q must be between 0 and 1')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            values = np.nanquantile(self.values, q, axis=1)
        return self._summary(values, name='quantile')

    # ------------------------------------------------------------------
    # trace-preserving transforms

    def _with_values(self, values: np.ndarray) -> typing.Self:
        values = np.asarray(values)
        if not np.issubdtype(values.dtype, np.floating) or values.dtype != self.dtype:
            values = values.astype(self.dtype)

        return self.__class__(
            _TracesData(
                values,
                self.sampling,
                self.start,
            ),
            self.meta,
        )

    def log10(self) -> typing.Self:
        with np.errstate(divide='ignore', invalid='ignore'):
            values = np.log10(self.values)
        return self._with_values(values)

    def power(self, exponent: float) -> typing.Self:
        exponent = float(exponent)
        if not np.isfinite(exponent):
            raise ValueError('exponent must be finite')

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            values = np.power(self.values, exponent)
        return self._with_values(values)

    def center(self) -> typing.Self:
        means = self.mean().to_numpy()
        return self._with_values(self.values - means[:, None])

    def zscore(self) -> typing.Self:
        means = self.mean().to_numpy()
        std = self.std().to_numpy()

        with np.errstate(divide='ignore', invalid='ignore'):
            values = (self.values - means[:, None]) / std[:, None]
        return self._with_values(values)

    def normalize_quantiles(
        self,
        qmin: float = 0.05,
        qmax: float = 0.95,
        *,
        win: Win | None = None,
    ) -> typing.Self:
        qmin = float(qmin)
        qmax = float(qmax)

        if not 0 <= qmin < qmax <= 1:
            raise ValueError('expected 0 <= qmin < qmax <= 1')

        reference = self if win is None else self.sel_time(win)
        if reference.n_samples == 0:
            raise ValueError('normalization window contains no samples')

        low = reference.quantile(qmin).to_numpy()
        high = reference.quantile(qmax).to_numpy()
        scale = high - low

        values = np.full(self.shape, np.nan, dtype=self.dtype)
        valid_scale = ~np.isnan(scale) & (scale != 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            values[valid_scale] = (
                self.values[valid_scale] - low[valid_scale, None]
            ) / scale[valid_scale, None]

        return self._with_values(values)

    # ------------------------------------------------------------------
    # rolling transforms

    def _rolling_bounds(
        self,
        window_ms: float,
        *,
        center: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        window_ms = float(window_ms)
        if not np.isfinite(window_ms) or window_ms <= 0:
            raise ValueError('window_ms must be finite and positive')

        window_samples = self.sampling.ms_to_samples(window_ms)
        if window_samples <= 0:
            raise ValueError('window_ms is shorter than one sample period')

        rounded = self.sampling.samples_to_ms(window_samples)
        if not np.isclose(window_ms, rounded):
            logger.warning(
                'Adjusting rolling window from %g ms to %g ms (%d samples)',
                window_ms,
                rounded,
                window_samples,
            )

        pos = np.arange(self.n_samples, dtype=np.intp)
        if center:
            left = (window_samples - 1) // 2
            right = window_samples - left
            start = pos - left
            stop = pos + right
        else:
            start = pos - window_samples + 1
            stop = pos + 1

        return (
            np.clip(start, 0, self.n_samples),
            np.clip(stop, 0, self.n_samples),
        )

    @staticmethod
    def _validate_min_valid(min_valid: int) -> int:
        min_valid = int(min_valid)
        if min_valid < 1:
            raise ValueError('min_valid must be at least 1')
        return min_valid

    def rolling_mean(
        self,
        window_ms: float,
        *,
        center: bool = True,
        min_valid: int = 1,
    ) -> typing.Self:
        min_valid = self._validate_min_valid(min_valid)
        start, stop = self._rolling_bounds(window_ms, center=center)

        valid = ~np.isnan(self.values)
        values = np.where(valid, self.values, 0)
        cumsum = np.pad(
            np.cumsum(values, axis=1, dtype=np.float64),
            ((0, 0), (1, 0)),
        )
        cumcount = np.pad(
            np.cumsum(valid, axis=1, dtype=np.int64),
            ((0, 0), (1, 0)),
        )

        total = cumsum[:, stop] - cumsum[:, start]
        count = cumcount[:, stop] - cumcount[:, start]
        output = np.full(self.shape, np.nan, dtype=np.float64)
        np.divide(
            total,
            count,
            out=output,
            where=count >= min_valid,
        )
        return self._with_values(output)

    def rolling_median(
        self,
        window_ms: float,
        *,
        center: bool = True,
        min_valid: int = 1,
    ) -> typing.Self:
        min_valid = self._validate_min_valid(min_valid)
        start, stop = self._rolling_bounds(window_ms, center=center)
        output = np.full(self.shape, np.nan, dtype=self.dtype)

        for i, (i0, i1) in enumerate(zip(start, stop, strict=True)):
            window = self.values[:, i0:i1]
            count = np.count_nonzero(~np.isnan(window), axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median = np.nanmedian(window, axis=1)
            output[count >= min_valid, i] = median[count >= min_valid]

        return self._with_values(output)

    def rolling_std(
        self,
        window_ms: float,
        *,
        center: bool = True,
        min_valid: int = 1,
    ) -> typing.Self:
        min_valid = self._validate_min_valid(min_valid)
        start, stop = self._rolling_bounds(window_ms, center=center)

        valid = ~np.isnan(self.values)
        values = np.where(valid, self.values, 0).astype(np.float64, copy=False)
        sums = np.pad(
            np.cumsum(values, axis=1),
            ((0, 0), (1, 0)),
        )
        sums2 = np.pad(
            np.cumsum(values * values, axis=1),
            ((0, 0), (1, 0)),
        )
        counts = np.pad(
            np.cumsum(valid, axis=1, dtype=np.int64),
            ((0, 0), (1, 0)),
        )

        total = sums[:, stop] - sums[:, start]
        total2 = sums2[:, stop] - sums2[:, start]
        count = counts[:, stop] - counts[:, start]

        output = np.full(self.shape, np.nan, dtype=np.float64)
        enough = count >= max(min_valid, 2)
        numerator = total2 - total * total / np.maximum(count, 1)
        variance = np.full(self.shape, np.nan, dtype=np.float64)
        np.divide(
            numerator,
            count - 1,
            out=variance,
            where=enough,
        )
        variance = np.maximum(variance, 0, where=~np.isnan(variance), out=variance)
        output[enough] = np.sqrt(variance[enough])

        return self._with_values(output)

    def rolling_zscore(
        self,
        window_ms: float,
        *,
        center: bool = True,
        min_valid: int = 1,
    ) -> typing.Self:
        mean = self.rolling_mean(
            window_ms,
            center=center,
            min_valid=min_valid,
        )
        std = self.rolling_std(
            window_ms,
            center=center,
            min_valid=min_valid,
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            values = (self.values - mean.values) / std.values
        return self._with_values(values)

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str = 'traces',
        overwrite: bool = False,
    ) -> None:
        """
        Store Traces in HDF5.

        Layout
        ------
        /<key>
            attrs:
                kind = 'traces'
                nocte_version = <current version>

            /meta
                trace metadata

            /data
                attrs:
                    sampling_rate_hz
                    start_ms

                two-dimensional floating-point dataset with shape
                (n_traces, n_samples)

        If ``overwrite`` is False, an existing collection root raises
        FileExistsError. If True, the complete existing subtree is removed
        before writing.
        """
        key = nocte.core.hdf.prepare_hdf_key(path, key, overwrite=overwrite)

        self.meta.to_hdf(path, key=f'{key}/meta', mode='a', format='fixed')

        nocte.core.hdf.write_hdf_collection_attrs(path, key, kind='traces')

        self._data.to_hdf(path, key=f'{key}/data')

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str = 'traces',
    ) -> typing.Self:
        """
        Load Traces previously stored with to_hdf().
        """
        key = nocte.core.hdf.check_hdf_collection_attrs(
            path, key, expected_kind='traces'
        )

        loaded_meta = pd.read_hdf(path, key=f'{key}/meta')

        if not isinstance(
            loaded_meta,
            pd.DataFrame,
        ):
            raise TypeError('stored Traces metadata must be a DataFrame')

        data = _TracesData.from_hdf(path, key=f'{key}/data')

        return cls(data, loaded_meta)
