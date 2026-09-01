"""Private numerical implementation for spectral analysis."""

from __future__ import annotations

import collections.abc
import typing

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.signal

import nocte.analysis._num_core
import nocte.core.sampling
import nocte.core.traces

Band = tuple[float, float]
Bands = collections.abc.Mapping[str, Band]

FloatArray = nocte.analysis._num_core.FloatArray
ComplexArray = nocte.analysis._num_core.ComplexArray
IntArray = nocte.analysis._num_core.IntArray

ROLLING_BATCH_SIZE = 32


class Butterworth(typing.NamedTuple):
    """Prepared zero-phase Butterworth filter in SOS form."""

    sos: FloatArray

    @staticmethod
    def order(
        order: int,
    ) -> int:
        if not isinstance(order, (int, np.integer)):
            raise TypeError('order must be an integer')

        order = int(order)

        if order <= 0:
            raise ValueError('order must be positive')

        return order

    @staticmethod
    def cutoff(
        cutoff: float,
        *,
        nyquist: float,
        name: str,
    ) -> float:
        cutoff = float(cutoff)

        if not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError(f'{name} must be finite and positive')

        if cutoff >= nyquist:
            raise ValueError(f'{name} must be below Nyquist frequency ({nyquist:g} Hz)')

        return cutoff

    @classmethod
    def build(
        cls,
        *,
        hz: float,
        cutoff: float | Band,
        order: int,
        btype: typing.Literal[
            'lowpass',
            'highpass',
            'bandpass',
        ],
    ) -> typing.Self:
        result = scipy.signal.butter(
            order,
            cutoff,
            btype=btype,
            fs=hz,
            output='sos',
        )

        sos = typing.cast(
            np.ndarray,
            result,
        )

        return cls(
            sos=np.ascontiguousarray(
                sos,
                dtype=np.float64,
            )
        )

    @classmethod
    def low_pass(
        cls,
        *,
        hz: float,
        cutoff: float,
        order: int,
    ) -> typing.Self:
        order = cls.order(order)
        cutoff = cls.cutoff(
            cutoff,
            nyquist=hz / 2.0,
            name='cutoff',
        )

        return cls.build(
            hz=hz,
            cutoff=cutoff,
            order=order,
            btype='lowpass',
        )

    @classmethod
    def high_pass(
        cls,
        *,
        hz: float,
        cutoff: float,
        order: int,
    ) -> typing.Self:
        order = cls.order(order)
        cutoff = cls.cutoff(
            cutoff,
            nyquist=hz / 2.0,
            name='cutoff',
        )

        return cls.build(
            hz=hz,
            cutoff=cutoff,
            order=order,
            btype='highpass',
        )

    @classmethod
    def band_pass(
        cls,
        *,
        hz: float,
        band: Band,
        order: int,
    ) -> typing.Self:
        if len(band) != 2:
            raise ValueError('band must contain exactly two frequencies')

        order = cls.order(order)
        nyquist = hz / 2.0

        low = cls.cutoff(
            band[0],
            nyquist=nyquist,
            name='band low bound',
        )
        high = cls.cutoff(
            band[1],
            nyquist=nyquist,
            name='band high bound',
        )

        if low >= high:
            raise ValueError('band must satisfy low < high')

        return cls.build(
            hz=hz,
            cutoff=(low, high),
            order=order,
            btype='bandpass',
        )

    def apply(
        self,
        traces: nocte.core.traces.Traces,
    ) -> nocte.core.traces.Traces:
        """Apply the prepared filter over each trace's finite support."""
        bounds = nocte.analysis._num_core.Bounds.from_traces(traces)

        values = np.full(
            traces.shape,
            np.nan,
            dtype=np.float64,
        )

        for position, (start, stop) in enumerate(
            zip(
                bounds.first,
                bounds.stop,
                strict=True,
            )
        ):
            if start == stop:
                continue

            try:
                values[
                    position,
                    start:stop,
                ] = scipy.signal.sosfiltfilt(
                    self.sos,
                    traces.values[
                        position,
                        start:stop,
                    ],
                )

            except ValueError as exc:
                trace_id = traces.index[position]
                raise ValueError(
                    f'trace {trace_id!r} is too short for the requested filter'
                ) from exc

        return nocte.analysis._num_core.traces_like(
            traces,
            values,
        )


class Analytic(typing.NamedTuple):
    """Complex analytic signal plus finite support of the source traces."""

    values: ComplexArray
    bounds: nocte.analysis._num_core.Bounds

    @classmethod
    def from_traces(
        cls,
        traces: nocte.core.traces.Traces,
    ) -> typing.Self:
        bounds = nocte.analysis._num_core.Bounds.from_traces(traces)

        values = np.full(
            traces.shape,
            np.nan + 1j * np.nan,
            dtype=np.complex128,
        )

        for position, (start, stop) in enumerate(
            zip(
                bounds.first,
                bounds.stop,
                strict=True,
            )
        ):
            if start == stop:
                continue

            values[
                position,
                start:stop,
            ] = cls.hilbert(
                traces.values[
                    position,
                    start:stop,
                ]
            )

        return cls(
            values=values,
            bounds=bounds,
        )

    @staticmethod
    def hilbert(
        values: np.ndarray,
    ) -> ComplexArray:
        result = scipy.signal.hilbert(values)

        return typing.cast(
            ComplexArray,
            result,
        )

    def phase(
        self,
        *,
        unwrap: bool,
    ) -> FloatArray:
        values = np.asarray(
            np.angle(self.values),
            dtype=np.float64,
        )

        if not unwrap:
            return values

        for position, (start, stop) in enumerate(
            zip(
                self.bounds.first,
                self.bounds.stop,
                strict=True,
            )
        ):
            if start == stop:
                continue

            values[
                position,
                start:stop,
            ] = np.unwrap(
                values[
                    position,
                    start:stop,
                ]
            )

        return values

    def amplitude(
        self,
    ) -> FloatArray:
        return np.asarray(
            np.abs(self.values),
            dtype=np.float64,
        )

    def frequency(
        self,
        hz: float,
    ) -> FloatArray:
        """Return instantaneous frequency in Hz."""
        values = np.full(
            self.values.shape,
            np.nan,
            dtype=np.float64,
        )

        scale = float(hz) / (2.0 * np.pi)

        for position, (start, stop) in enumerate(
            zip(
                self.bounds.first,
                self.bounds.stop,
                strict=True,
            )
        ):
            if stop - start < 2:
                continue

            phase = np.unwrap(
                np.angle(
                    self.values[
                        position,
                        start:stop,
                    ]
                )
            )

            values[
                position,
                start + 1 : stop,
            ] = np.diff(phase) * scale

        return values


class Welch(typing.NamedTuple):
    """Prepared Welch PSD definition with its fixed frequency grid."""

    hz: float
    nperseg: int
    frequency: FloatArray

    @classmethod
    def from_traces(
        cls,
        traces: nocte.core.traces.Traces,
        *,
        segment: float,
    ) -> typing.Self:
        nperseg = nocte.analysis._num_core.duration_samples(
            traces.sampling,
            segment,
            desc='segment',
        )

        if nperseg > traces.n_samples:
            raise ValueError('segment exceeds trace duration')

        frequency = np.fft.rfftfreq(
            nperseg,
            d=1.0 / traces.hz,
        )

        return cls(
            hz=traces.hz,
            nperseg=nperseg,
            frequency=np.asarray(
                frequency,
                dtype=np.float64,
            ),
        )

    def power(
        self,
        values: np.ndarray,
    ) -> FloatArray:
        """Calculate Welch PSD along the final array axis."""
        if values.shape[-1] < self.nperseg:
            raise ValueError('values contain fewer samples than segment')

        _, _power = scipy.signal.welch(
            values,
            fs=self.hz,
            window='hann',
            nperseg=self.nperseg,
            noverlap=self.nperseg // 2,
            nfft=self.nperseg,
            detrend='constant',
            return_onesided=True,
            scaling='density',
            axis=-1,
            average='mean',
        )

        return np.asarray(
            _power,
            dtype=np.float64,
        )

    def power_traces(
        self,
        traces: nocte.core.traces.Traces,
        bounds: nocte.analysis._num_core.Bounds,
    ) -> FloatArray:
        """Calculate one PSD per trace over its complete finite support."""
        if len(traces) == 0:
            return np.empty(
                (
                    0,
                    len(self.frequency),
                ),
                dtype=np.float64,
            )

        if bounds.are_full(traces.n_samples):
            return self.power(traces.values)

        power = np.full(
            (
                len(traces),
                len(self.frequency),
            ),
            np.nan,
            dtype=np.float64,
        )

        for position, (start, stop) in enumerate(
            zip(
                bounds.first,
                bounds.stop,
                strict=True,
            )
        ):
            if start == stop:
                continue

            if stop - start < self.nperseg:
                trace_id = traces.index[position]
                raise ValueError(
                    f'trace {trace_id!r} has fewer finite samples than segment'
                )

            power[position] = self.power(
                traces.values[
                    position,
                    start:stop,
                ]
            )

        return power


class BandPlan(typing.NamedTuple):
    """Validated named bands mapped onto one Welch frequency grid."""

    names: tuple[str, ...]
    starts: IntArray
    stops: IntArray
    frequency: FloatArray

    @classmethod
    def from_bands(
        cls,
        bands: Bands,
        welch: Welch,
    ) -> typing.Self:
        if not isinstance(
            bands,
            collections.abc.Mapping,
        ):
            raise TypeError('bands must be a mapping of name -> (low_hz, high_hz)')

        if not bands:
            raise ValueError('bands cannot be empty')

        names: list[str] = []
        starts: list[int] = []
        stops: list[int] = []

        nyquist = welch.hz / 2.0

        for name, band in bands.items():
            if not isinstance(name, str) or not name:
                raise ValueError('band names must be non-empty strings')

            if len(band) != 2:
                raise ValueError(f'band {name!r} must contain exactly two frequencies')

            low = float(band[0])
            high = float(band[1])

            if not np.isfinite(low) or not np.isfinite(high):
                raise ValueError(f'band {name!r} bounds must be finite')

            if low < 0 or high <= low:
                raise ValueError(f'band {name!r} must satisfy 0 <= low < high')

            if high > nyquist:
                raise ValueError(
                    f'band {name!r} extends above Nyquist frequency ({nyquist:g} Hz)'
                )

            start = int(
                np.searchsorted(
                    welch.frequency,
                    low,
                    side='left',
                )
            )
            stop = int(
                np.searchsorted(
                    welch.frequency,
                    high,
                    side='right',
                )
            )

            if stop - start < 2:
                raise ValueError(
                    f'band {name!r} contains fewer than two Welch frequency bins'
                )

            names.append(name)
            starts.append(start)
            stops.append(stop)

        return cls(
            names=tuple(names),
            starts=np.asarray(starts, dtype=np.intp),
            stops=np.asarray(stops, dtype=np.intp),
            frequency=welch.frequency,
        )

    def integrate(
        self,
        power: FloatArray,
    ) -> FloatArray:
        """Integrate a PSD array over each band along its final axis."""
        if power.shape[-1] != len(self.frequency):
            raise ValueError('power does not match the prepared frequency grid')

        result = np.empty(
            power.shape[:-1] + (len(self.names),),
            dtype=np.float64,
        )

        for band, (start, stop) in enumerate(
            zip(
                self.starts,
                self.stops,
                strict=True,
            )
        ):
            integral = scipy.integrate.simpson(
                power[
                    ...,
                    start:stop,
                ],
                x=self.frequency[start:stop],
                axis=-1,
            )

            result[..., band] = np.asarray(
                integral,
                dtype=np.float64,
            )

        return result


class Rolling(typing.NamedTuple):
    """Fixed rolling geometry over one source sampling grid."""

    window_samples: int
    step_samples: int
    grid: nocte.core.sampling.TimeGrid

    @classmethod
    def from_traces(
        cls,
        traces: nocte.core.traces.Traces,
        *,
        window: float,
        step: float,
    ) -> typing.Self:
        window_samples = nocte.analysis._num_core.duration_samples(
            traces.sampling,
            window,
            desc='window',
        )
        step_samples = nocte.analysis._num_core.duration_samples(
            traces.sampling,
            step,
            desc='step',
        )

        if window_samples > traces.n_samples:
            raise ValueError('window exceeds trace duration')

        n_times = (traces.n_samples - window_samples) // step_samples + 1
        center_offset = traces.sampling.samples_to_ms(window_samples) / 2.0

        grid = nocte.core.sampling.TimeGrid(
            sampling=traces.sampling.strided(step_samples),
            start=traces.start + center_offset,
            n_samples=n_times,
        )

        return cls(
            window_samples=window_samples,
            step_samples=step_samples,
            grid=grid,
        )

    def windows(
        self,
        traces: nocte.core.traces.Traces,
    ) -> np.ndarray:
        """Return a zero-copy view of rolling source windows."""
        nocte.analysis._num_core.Bounds.from_traces(traces)

        windows = np.lib.stride_tricks.sliding_window_view(
            traces.values,
            window_shape=self.window_samples,
            axis=1,
        )

        return windows[
            :,
            :: self.step_samples,
            :,
        ]

    def validate_welch(
        self,
        welch: Welch,
    ) -> None:
        if welch.nperseg > self.window_samples:
            raise ValueError('segment cannot be longer than window')

    def welch(
        self,
        traces: nocte.core.traces.Traces,
        welch: Welch,
    ) -> FloatArray:
        """Calculate rolling PSD without copying source windows."""
        self.validate_welch(welch)
        windows = self.windows(traces)

        result = np.empty(
            (
                len(traces),
                len(welch.frequency),
                self.grid.n_samples,
            ),
            dtype=np.float64,
        )

        if len(traces) == 0:
            return result

        for start in range(
            0,
            self.grid.n_samples,
            ROLLING_BATCH_SIZE,
        ):
            stop = min(
                start + ROLLING_BATCH_SIZE,
                self.grid.n_samples,
            )

            power = welch.power(
                windows[
                    :,
                    start:stop,
                    :,
                ]
            )

            result[
                :,
                :,
                start:stop,
            ] = np.moveaxis(
                power,
                -1,
                1,
            )

        return result

    def band_power(
        self,
        traces: nocte.core.traces.Traces,
        welch: Welch,
        bands: BandPlan,
    ) -> FloatArray:
        """Calculate rolling band power without storing the full PSD cube."""
        self.validate_welch(welch)
        windows = self.windows(traces)

        result = np.empty(
            (
                len(traces),
                len(bands.names),
                self.grid.n_samples,
            ),
            dtype=np.float64,
        )

        if len(traces) == 0:
            return result

        for start in range(
            0,
            self.grid.n_samples,
            ROLLING_BATCH_SIZE,
        ):
            stop = min(
                start + ROLLING_BATCH_SIZE,
                self.grid.n_samples,
            )

            power = welch.power(
                windows[
                    :,
                    start:stop,
                    :,
                ]
            )
            integrated = bands.integrate(power)

            result[
                :,
                :,
                start:stop,
            ] = np.moveaxis(
                integrated,
                -1,
                1,
            )

        return result


def low_pass(
    traces: nocte.core.traces.Traces,
    cutoff: float,
    *,
    order: int,
) -> nocte.core.traces.Traces:
    filter_ = Butterworth.low_pass(
        hz=traces.hz,
        cutoff=cutoff,
        order=order,
    )

    return filter_.apply(traces)


def high_pass(
    traces: nocte.core.traces.Traces,
    cutoff: float,
    *,
    order: int,
) -> nocte.core.traces.Traces:
    filter_ = Butterworth.high_pass(
        hz=traces.hz,
        cutoff=cutoff,
        order=order,
    )

    return filter_.apply(traces)


def band_pass(
    traces: nocte.core.traces.Traces,
    band: Band,
    *,
    order: int,
) -> nocte.core.traces.Traces:
    filter_ = Butterworth.band_pass(
        hz=traces.hz,
        band=band,
        order=order,
    )

    return filter_.apply(traces)


def hilbert(
    traces: nocte.core.traces.Traces,
) -> ComplexArray:
    return Analytic.from_traces(traces).values


def hilbert_phase(
    traces: nocte.core.traces.Traces,
    *,
    unwrap: bool,
) -> nocte.core.traces.Traces:
    analytic = Analytic.from_traces(traces)

    return nocte.analysis._num_core.traces_like(
        traces,
        analytic.phase(unwrap=unwrap),
    )


def hilbert_amplitude(
    traces: nocte.core.traces.Traces,
) -> nocte.core.traces.Traces:
    analytic = Analytic.from_traces(traces)

    return nocte.analysis._num_core.traces_like(
        traces,
        analytic.amplitude(),
    )


def instantaneous_frequency(
    traces: nocte.core.traces.Traces,
) -> nocte.core.traces.Traces:
    analytic = Analytic.from_traces(traces)

    return nocte.analysis._num_core.traces_like(
        traces,
        analytic.frequency(traces.hz),
    )


def welch(
    traces: nocte.core.traces.Traces,
    *,
    segment: float,
    db: bool,
) -> pd.DataFrame:
    bounds = nocte.analysis._num_core.Bounds.from_traces(traces)
    estimate = Welch.from_traces(
        traces,
        segment=segment,
    )

    values = estimate.power_traces(
        traces,
        bounds,
    )

    if db:
        values = nocte.analysis._num_core.to_db(values)

    return pd.DataFrame(
        values,
        index=traces.index.copy(),
        columns=pd.Index(
            estimate.frequency,
            name='frequency_hz',
        ),
    )


def band_power(
    traces: nocte.core.traces.Traces,
    bands: Bands,
    *,
    segment: float,
    db: bool,
) -> pd.DataFrame:
    bounds = nocte.analysis._num_core.Bounds.from_traces(traces)
    estimate = Welch.from_traces(
        traces,
        segment=segment,
    )
    plan = BandPlan.from_bands(
        bands,
        estimate,
    )

    power = estimate.power_traces(
        traces,
        bounds,
    )
    values = plan.integrate(power)

    if db:
        values = nocte.analysis._num_core.to_db(values)

    return pd.DataFrame(
        values,
        index=traces.index.copy(),
        columns=pd.Index(
            plan.names,
            name='band',
        ),
    )


def welch_rolling(
    traces: nocte.core.traces.Traces,
    *,
    window: float,
    step: float,
    segment: float,
    db: bool,
) -> nocte.core.traces.Traces:
    estimate = Welch.from_traces(
        traces,
        segment=segment,
    )
    rolling = Rolling.from_traces(
        traces,
        window=window,
        step=step,
    )

    values = rolling.welch(
        traces,
        estimate,
    )

    if db:
        values = nocte.analysis._num_core.to_db(values)

    return nocte.analysis._num_core.feature_traces(
        values,
        rolling.grid,
        source_meta=traces.meta,
        source_ids=traces.index.to_numpy(copy=False),
        source_name=traces.name,
        feature_name='frequency_hz',
        features=estimate.frequency,
        result_name='power',
    )


def band_power_rolling(
    traces: nocte.core.traces.Traces,
    bands: Bands,
    *,
    window: float,
    step: float,
    segment: float,
    db: bool,
) -> nocte.core.traces.Traces:
    estimate = Welch.from_traces(
        traces,
        segment=segment,
    )
    plan = BandPlan.from_bands(
        bands,
        estimate,
    )
    rolling = Rolling.from_traces(
        traces,
        window=window,
        step=step,
    )

    values = rolling.band_power(
        traces,
        estimate,
        plan,
    )

    if db:
        values = nocte.analysis._num_core.to_db(values)

    return nocte.analysis._num_core.feature_traces(
        values,
        rolling.grid,
        source_meta=traces.meta,
        source_ids=traces.index.to_numpy(copy=False),
        source_name=traces.name,
        feature_name='band',
        features=plan.names,
        result_name='power',
    )
