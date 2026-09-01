from __future__ import annotations

import dataclasses
import logging
import typing
import warnings

import numpy as np

from nocte._core.time import S_TO_MS

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class SamplingRate:
    """Sampling rate and conversions between milliseconds and sample indices."""

    rate: float

    def __post_init__(self) -> None:
        rate = float(self.rate)

        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('Sampling rate must be finite and positive')

        object.__setattr__(
            self,
            'rate',
            rate,
        )

    @classmethod
    def from_period_ms(
        cls,
        period_ms: float,
    ) -> typing.Self:
        """Build a sampling rate from its sampling period in milliseconds."""
        period_ms = float(period_ms)

        if not np.isfinite(period_ms) or period_ms <= 0:
            raise ValueError('Sampling period must be finite and positive')

        return cls(S_TO_MS / period_ms)

    @property
    def period_ms(self) -> float:
        """Sampling period in milliseconds."""
        return S_TO_MS / self.rate

    def _sample_positions(
        self,
        times: float | np.ndarray,
        *,
        anchor: float = 0.0,
    ) -> np.ndarray:
        """Express times as floating-point sample offsets from anchor."""
        times = np.asarray(
            times,
            dtype=float,
        )
        anchor = float(anchor)

        if not np.all(np.isfinite(times)):
            raise ValueError('times must be finite')

        if not np.isfinite(anchor):
            raise ValueError('anchor must be finite')

        return (times - anchor) / self.period_ms

    def _snap_sample_positions(
        self,
        samples: np.ndarray,
        *,
        rtol: float,
        atol: float,
    ) -> np.ndarray:
        """
        Snap values sufficiently close to integer sample positions.

        `rtol` is interpreted relative to one sampling period rather than the
        absolute elapsed sample index.
        """
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError('rtol must be finite and non-negative')

        if not np.isfinite(atol) or atol < 0:
            raise ValueError('atol must be finite and non-negative')

        nearest = np.rint(samples)

        tolerance_samples = rtol + atol / self.period_ms

        return np.where(
            np.abs(samples - nearest) <= tolerance_samples,
            nearest,
            samples,
        )

    @typing.overload
    def ms_to_samples_exact(
        self,
        times: float,
        *,
        anchor: float = 0.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        desc: str | None = None,
    ) -> int: ...

    @typing.overload
    def ms_to_samples_exact(
        self,
        times: np.ndarray,
        *,
        anchor: float = 0.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        desc: str | None = None,
    ) -> np.ndarray: ...

    def ms_to_samples_exact(
        self,
        times: float | np.ndarray,
        *,
        anchor: float = 0.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        desc: str | None = None,
    ) -> int | np.ndarray:
        """
        Convert times to integer sample offsets from anchor.

        Raise if any time is not aligned to the sampling grid. `rtol` is
        interpreted relative to one sampling period rather than absolute
        elapsed time.
        """
        desc = desc or 'time'
        scalar = np.ndim(times) == 0

        samples = self._sample_positions(
            times,
            anchor=anchor,
        )
        snapped = self._snap_sample_positions(
            samples,
            rtol=rtol,
            atol=atol,
        )

        if not np.all(snapped == np.rint(snapped)):
            raise ValueError(f'{desc} are not aligned to the sampling grid')

        result = np.rint(snapped).astype(np.intp)

        if scalar:
            return int(result)

        return result

    @typing.overload
    def ms_to_samples_ceil(
        self,
        times: float,
        *,
        anchor: float = 0.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> int: ...

    @typing.overload
    def ms_to_samples_ceil(
        self,
        times: np.ndarray,
        *,
        anchor: float = 0.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> np.ndarray: ...

    def ms_to_samples_ceil(
        self,
        times: float | np.ndarray,
        *,
        anchor: float = 0.0,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> int | np.ndarray:
        """
        Convert times to the first sample offsets at or after each time.

        Values sufficiently close to an exact sample position are first
        snapped to that position to avoid floating-point boundary errors.
        """
        scalar = np.ndim(times) == 0

        samples = self._sample_positions(
            times,
            anchor=anchor,
        )
        samples = self._snap_sample_positions(
            samples,
            rtol=rtol,
            atol=atol,
        )

        result = np.ceil(samples).astype(np.intp)

        if scalar:
            return int(result)

        return result

    @typing.overload
    def ms_to_samples(
        self,
        time_ms: float,
        *,
        anchor: float = 0.0,
    ) -> int: ...

    @typing.overload
    def ms_to_samples(
        self,
        time_ms: np.ndarray,
        *,
        anchor: float = 0.0,
    ) -> np.ndarray: ...

    def ms_to_samples(
        self,
        time_ms: float | np.ndarray,
        *,
        anchor: float = 0.0,
    ) -> int | np.ndarray:
        """Convert milliseconds to the nearest sample offsets from anchor."""
        samples = np.rint(
            self._sample_positions(
                time_ms,
                anchor=anchor,
            )
        ).astype(np.intp)

        if samples.ndim == 0:
            return int(samples)

        return samples

    @typing.overload
    def samples_to_ms(
        self,
        samples: int,
        *,
        anchor: float = 0.0,
    ) -> float: ...

    @typing.overload
    def samples_to_ms(
        self,
        samples: np.ndarray,
        *,
        anchor: float = 0.0,
    ) -> np.ndarray: ...

    def samples_to_ms(
        self,
        samples: int | np.ndarray,
        *,
        anchor: float = 0.0,
    ) -> float | np.ndarray:
        """Convert sample offsets to milliseconds relative to anchor."""
        anchor = float(anchor)

        if not np.isfinite(anchor):
            raise ValueError('anchor must be finite')

        time_ms = (
            anchor
            + np.asarray(
                samples,
                dtype=float,
            )
            * self.period_ms
        )

        if time_ms.ndim == 0:
            return float(time_ms)

        return time_ms

    def strided(
        self,
        stride: int,
    ) -> typing.Self:
        """Return the sampling rate obtained by taking every `stride`th sample."""
        if not isinstance(
            stride,
            (int, np.integer),
        ):
            raise TypeError('stride must be an integer')

        stride = int(stride)

        if stride <= 0:
            raise ValueError('stride must be positive')

        return self.__class__(self.rate / stride)

    def _validate_target_hz(
        self,
        target_hz: float,
    ) -> float:
        target_hz = float(target_hz)

        if not np.isfinite(target_hz) or target_hz <= 0:
            raise ValueError('Target sampling rate must be finite and positive')

        if target_hz > self.rate:
            raise ValueError(
                f'Target sampling rate ({target_hz} Hz) cannot exceed '
                f'source sampling rate ({self.rate} Hz)'
            )

        return target_hz

    def stride_for(
        self,
        target_hz: float,
    ) -> int:
        """
        Return the integer stride closest to a target sampling rate.

        The resulting sampling rate may differ slightly from `target_hz`.
        """
        target_hz = self._validate_target_hz(target_hz)

        return int(np.round(self.rate / target_hz))

    def match_hz(
        self,
        target_hz: float,
        *,
        thresh: float | None = None,
    ) -> float:
        """Return the sampling rate corresponding to the nearest integer stride."""
        target_hz = self._validate_target_hz(target_hz)

        stride = self.stride_for(target_hz)
        matched_hz = self.strided(stride).rate

        if thresh is None:
            valid = np.isclose(
                matched_hz,
                target_hz,
            )
        else:
            thresh = float(thresh)

            if not np.isfinite(thresh) or thresh < 0:
                raise ValueError('Threshold must be finite and non-negative')

            valid = abs(matched_hz - target_hz) < thresh

        if not valid:
            logger.warning(
                'Adjusting target sampling rate from %g Hz to %g Hz '
                'to obtain an integer stride of %d from %g Hz',
                target_hz,
                matched_hz,
                stride,
                self.rate,
            )

        return matched_hz

    def check_stride(
        self,
        target_hz: float,
    ) -> bool:
        """Return whether a target sampling rate corresponds to an integer stride."""
        target_hz = self._validate_target_hz(target_hz)

        return bool(
            np.isclose(
                self.stride_for(target_hz),
                self.rate / target_hz,
            )
        )

    def assert_stride(
        self,
        target_hz: float,
        numerator_name: str = 'sampling_hz',
        denominator_name: str = 'target_hz',
    ) -> None:
        """Raise if a target sampling rate does not correspond to an integer stride."""
        if not self.check_stride(target_hz):
            raise AssertionError(
                f'Expected {denominator_name} ({target_hz}) to be a divisor of '
                f'{numerator_name} ({self.rate})'
            )

    def round_to_period(
        self,
        value_ms: float,
        *,
        desc: str | None = None,
    ) -> float:
        """Round a millisecond value to the nearest sampling period."""
        value_ms = float(value_ms)

        if not np.isfinite(value_ms):
            raise ValueError('value_ms must be finite')

        rounded = self.samples_to_ms(self.ms_to_samples(value_ms))

        if desc is not None and not np.isclose(
            rounded,
            value_ms,
        ):
            logger.warning(
                'Adjusting %s from %g ms to %g ms',
                desc,
                value_ms,
                rounded,
            )

        return rounded

    def _repr_html_(
        self,
    ) -> str:
        rate = f'{self.rate:.3g}'
        period = f'{self.period_ms:.3g}'

        return f'{rate} Hz ({period} ms)'


@dataclasses.dataclass(
    frozen=True,
    slots=True,
)
class TimeGrid:
    """Finite regular sequence of temporal samples."""

    sampling: SamplingRate
    start: float
    n_samples: int

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.sampling,
            SamplingRate,
        ):
            raise TypeError('sampling must be a SamplingRate')

        start = float(self.start)

        if not np.isfinite(start):
            raise ValueError('start must be finite')

        if not isinstance(
            self.n_samples,
            (int, np.integer),
        ):
            raise TypeError('n_samples must be an integer')

        n_samples = int(self.n_samples)

        if n_samples < 0:
            raise ValueError('n_samples must be non-negative')

        object.__setattr__(
            self,
            'start',
            start,
        )
        object.__setattr__(
            self,
            'n_samples',
            n_samples,
        )

    @classmethod
    def from_times(
        cls,
        times: np.ndarray,
        *,
        rtol: float = 1e-7,
        atol: float = 1e-9,
        adjustment_atol: float = 1e-6,
        adjustment: typing.Literal[
            'warn',
            'raise',
        ] = 'warn',
    ) -> typing.Self:
        """
        Construct a regular grid from explicit sample times.

        The input times must be finite, strictly increasing, and approximately
        regularly spaced according to `rtol` and `atol`.

        The inferred period is converted through `SamplingRate` and the input
        coordinates are then compared with the resulting canonical grid. If
        the maximum adjustment exceeds `adjustment_atol`, either warn or raise
        according to `adjustment`.
        """
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError('rtol must be finite and non-negative')

        if not np.isfinite(atol) or atol < 0:
            raise ValueError('atol must be finite and non-negative')

        if not np.isfinite(adjustment_atol) or adjustment_atol < 0:
            raise ValueError('adjustment_atol must be finite and non-negative')

        if adjustment not in (
            'warn',
            'raise',
        ):
            raise ValueError("adjustment must be either 'warn' or 'raise'")

        times = np.asarray(
            times,
            dtype=float,
        )

        if times.ndim != 1:
            raise ValueError('times must be one-dimensional')

        if times.size < 2:
            raise ValueError('at least two times are required to infer a grid')

        if not np.all(np.isfinite(times)):
            raise ValueError('times must be finite')

        diffs = np.diff(times)

        if np.any(diffs <= 0):
            raise ValueError('times must be strictly increasing')

        period = float(np.median(diffs))

        if not np.allclose(
            diffs,
            period,
            rtol=rtol,
            atol=atol,
        ):
            max_error = float(np.max(np.abs(diffs - period)))

            raise ValueError(
                'times do not define a regular grid: '
                f'maximum interval deviation is {max_error:g} ms'
            )

        grid = cls(
            sampling=(SamplingRate.from_period_ms(period)),
            start=float(times[0]),
            n_samples=(times.size),
        )

        max_adjustment = float(np.max(np.abs(times - grid.times)))

        if max_adjustment > adjustment_atol:
            message = (
                'times require adjustment to the canonical sampling grid: '
                f'maximum adjustment is {max_adjustment:g} ms'
            )

            if adjustment == 'raise':
                raise ValueError(message)

            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )

        return grid

    @classmethod
    def from_aligned_bounds(
        cls,
        *,
        sampling: SamplingRate,
        start: float,
        stop: float,
        anchor: float,
        rtol: float = 1e-7,
        atol: float = 1e-9,
    ) -> typing.Self:
        """
        Construct a regular grid phase-locked to `anchor`.

        Grid coordinates are

            anchor + k * period

        and include every such coordinate in the half-open interval
        [start, stop).
        """
        if not isinstance(
            sampling,
            SamplingRate,
        ):
            raise TypeError('sampling must be a SamplingRate')

        start = float(start)
        stop = float(stop)
        anchor = float(anchor)

        if not np.all(
            np.isfinite(
                [
                    start,
                    stop,
                    anchor,
                ]
            )
        ):
            raise ValueError('start, stop, and anchor must be finite')

        if stop < start:
            raise ValueError('stop must not precede start')

        first = sampling.ms_to_samples_ceil(
            start,
            anchor=anchor,
            rtol=rtol,
            atol=atol,
        )
        last = sampling.ms_to_samples_ceil(
            stop,
            anchor=anchor,
            rtol=rtol,
            atol=atol,
        )

        return cls(
            sampling=sampling,
            start=sampling.samples_to_ms(
                first,
                anchor=anchor,
            ),
            n_samples=(last - first),
        )

    @classmethod
    def from_bounds(
        cls,
        *,
        sampling: SamplingRate,
        start: float,
        stop: float,
    ) -> typing.Self:
        """
        Construct the regular grid covering the half-open interval
        [start, stop).
        """
        return cls.from_aligned_bounds(
            sampling=sampling,
            start=start,
            stop=stop,
            anchor=start,
        )

    @classmethod
    def from_hz_bounds(
        cls,
        *,
        hz: float,
        start: float,
        stop: float,
    ) -> typing.Self:
        """Construct a grid from a sampling rate in Hz and temporal bounds."""
        return cls.from_bounds(
            sampling=SamplingRate(hz),
            start=start,
            stop=stop,
        )

    @classmethod
    def from_start_last(
        cls,
        *,
        sampling: SamplingRate,
        start: float,
        last: float,
        rtol: float = 1e-10,
        atol: float = 1e-10,
    ) -> typing.Self:
        """Construct a grid from its first coordinate through `last`, inclusive."""
        if not isinstance(sampling, SamplingRate):
            raise TypeError('sampling must be a SamplingRate')

        start = float(start)
        last = float(last)

        if not np.isfinite(start) or not np.isfinite(last):
            raise ValueError('start and last must be finite')

        if last < start:
            return cls(sampling=sampling, start=start, n_samples=0)

        position = (last - start) / sampling.period_ms
        position = sampling._snap_sample_positions(
            np.asarray(position),
            rtol=rtol,
            atol=atol,
        )

        return cls(
            sampling=sampling,
            start=start,
            n_samples=int(np.floor(position)) + 1,
        )

    @property
    def stop(
        self,
    ) -> float:
        """Exclusive temporal boundary of the grid."""
        return self.sampling.samples_to_ms(
            self.n_samples,
            anchor=self.start,
        )

    @property
    def last(
        self,
    ) -> float | None:
        """Time of the final sample, or None for an empty grid."""
        if self.n_samples == 0:
            return None

        return self.sampling.samples_to_ms(
            self.n_samples - 1,
            anchor=self.start,
        )

    @property
    def times(
        self,
    ) -> np.ndarray:
        """Times of all samples in the grid."""
        return self.sampling.samples_to_ms(
            np.arange(
                self.n_samples,
                dtype=np.intp,
            ),
            anchor=self.start,
        )

    def shift_time(
        self,
        by: float,
    ) -> typing.Self:
        """Shift all grid coordinates by a constant amount."""
        by = float(by)

        if not np.isfinite(by):
            raise ValueError('time shift must be finite')

        return self.__class__(
            sampling=self.sampling,
            start=self.start + by,
            n_samples=self.n_samples,
        )

    def _sample_geometry(
        self,
        sampling: SamplingRate,
        *,
        anchor: float = 0.0,
    ) -> tuple[
        int,
        int,
    ]:
        """
        Return this grid's start and stride on another sampling lattice.
        """
        if not isinstance(
            sampling,
            SamplingRate,
        ):
            raise TypeError('sampling must be a SamplingRate')

        start = sampling.ms_to_samples_exact(
            self.start,
            anchor=anchor,
            desc='grid start',
        )

        step = sampling.ms_to_samples_exact(
            self.sampling.period_ms,
            desc='grid sampling period',
        )

        if step <= 0:
            raise ValueError(
                'grid sampling period must span at least one target sample'
            )

        return (
            start,
            step,
        )

    def align_to(
        self,
        sampling: SamplingRate,
        *,
        anchor: float = 0.0,
    ) -> typing.Self:
        """
        Return this grid canonicalized to another sampling grid.

        The grid start and sampling period must correspond to integer sample
        offsets of `sampling`, phase-locked to `anchor`.
        """
        start, step = self._sample_geometry(
            sampling,
            anchor=anchor,
        )

        return self.__class__(
            sampling=sampling.strided(step),
            start=sampling.samples_to_ms(
                start,
                anchor=anchor,
            ),
            n_samples=self.n_samples,
        )

    def sample_offsets(
        self,
        sampling: SamplingRate,
        *,
        anchor: float = 0.0,
    ) -> np.ndarray:
        """
        Return this grid's coordinates as integer offsets on another grid.
        """
        start, step = self._sample_geometry(
            sampling,
            anchor=anchor,
        )

        return start + step * np.arange(
            self.n_samples,
            dtype=np.intp,
        )
