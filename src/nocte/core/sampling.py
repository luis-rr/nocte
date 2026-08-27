import dataclasses
import logging
import typing

import numpy as np

import nocte.core.time

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class SamplingRate:
    """Sampling rate and conversions between milliseconds and sample indices."""

    rate: float

    def __post_init__(self) -> None:
        rate = float(self.rate)

        if not np.isfinite(rate) or rate <= 0:
            raise ValueError('Sampling rate must be finite and positive')

        object.__setattr__(self, 'rate', rate)

    @classmethod
    def from_period_ms(cls, period_ms: float) -> typing.Self:
        """Build a sampling rate from its sampling period in milliseconds."""
        period_ms = float(period_ms)

        if not np.isfinite(period_ms) or period_ms <= 0:
            raise ValueError('Sampling period must be finite and positive')

        return cls(nocte.core.time.S_TO_MS / period_ms)

    @property
    def period_ms(self) -> float:
        """Sampling period in milliseconds."""
        return nocte.core.time.S_TO_MS / self.rate

    def _validate_target_hz(self, target_hz: float) -> float:
        target_hz = float(target_hz)

        if not np.isfinite(target_hz) or target_hz <= 0:
            raise ValueError('Target sampling rate must be finite and positive')

        if target_hz > self.rate:
            raise ValueError(
                f'Target sampling rate ({target_hz} Hz) cannot exceed '
                f'source sampling rate ({self.rate} Hz)'
            )

        return target_hz

    def stride_for(self, target_hz: float) -> int:
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
        matched_hz = self.rate / stride

        if thresh is None:
            valid = np.isclose(matched_hz, target_hz)
        else:
            if thresh < 0:
                raise ValueError('Threshold must be non-negative')

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

    def check_stride(self, target_hz: float) -> bool:
        """Return whether a target sampling rate corresponds to an integer stride."""
        target_hz = self._validate_target_hz(target_hz)
        return bool(np.isclose(self.stride_for(target_hz), self.rate / target_hz))

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

    @typing.overload
    def ms_to_samples(self, time_ms: float) -> int: ...

    @typing.overload
    def ms_to_samples(self, time_ms: np.ndarray) -> np.ndarray: ...

    def ms_to_samples(
        self,
        time_ms: float | np.ndarray,
    ) -> int | np.ndarray:
        """Convert milliseconds to the nearest sample indices."""
        samples = np.round(
            np.asarray(time_ms, dtype=float) * self.rate * nocte.core.time.MS_TO_S
        ).astype(np.intp)

        if samples.ndim == 0:
            return int(samples.item())

        return samples

    @typing.overload
    def samples_to_ms(self, samples: int) -> float: ...

    @typing.overload
    def samples_to_ms(self, samples: np.ndarray) -> np.ndarray: ...

    def samples_to_ms(
        self,
        samples: int | np.ndarray,
    ) -> float | np.ndarray:
        """Convert sample indices to milliseconds."""
        time_ms = np.asarray(samples, dtype=float) / self.rate * nocte.core.time.S_TO_MS

        if time_ms.ndim == 0:
            return float(time_ms.item())

        return time_ms

    def round_to_period(
        self,
        value_ms: float,
        *,
        desc: str | None = None,
    ) -> float:
        """Round a millisecond value to the nearest sampling period."""
        value_ms = float(value_ms)
        rounded = float(np.round(value_ms / self.period_ms) * self.period_ms)

        if desc is not None and not np.isclose(rounded, value_ms):
            logger.warning(
                'Adjusting %s from %g ms to %g ms',
                desc,
                value_ms,
                rounded,
            )

        return rounded
