"""Point-event temporal collection."""

from __future__ import annotations

import collections.abc
import pathlib
import typing
import warnings

import h5py
import numpy as np
import pandas as pd

import nocte.core._point_process
import nocte.core.hdf
import nocte.core.time
import nocte.core.windows

TimeArrayLike = float | collections.abc.Sequence[float] | np.ndarray | pd.Index
TimesLike = TimeArrayLike | pd.Series
EventValue = float | collections.abc.Sequence[float] | np.ndarray | pd.Series


class _EventsData:
    """Internal immutable storage for point-event times."""

    def __init__(self, values: np.ndarray):
        values = np.array(values, dtype=float, order='C', copy=True)

        if values.ndim != 1:
            raise ValueError('Event times must be one-dimensional')

        if not np.isfinite(values).all():
            raise ValueError('Event times must be finite')

        values.flags.writeable = False
        self.values = values

    @classmethod
    def from_times(cls, times: TimeArrayLike) -> typing.Self:
        values = np.asarray(times, dtype=float)

        if values.ndim == 0:
            values = values.reshape(1)
        elif values.ndim != 1:
            raise ValueError('Event times must be scalar or one-dimensional')

        return cls(values)

    def __len__(self) -> int:
        return len(self.values)

    def sel_pos(self, positions: np.ndarray) -> typing.Self:
        return self.__class__(self.values[positions])

    def get_pos(self, position: int) -> float:
        return float(self.values[position])

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        """Store event times as one HDF5 dataset."""
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='a') as file:
            if key in file:
                raise FileExistsError(f'HDF5 key {key!r} already exists')

            file.create_dataset(
                key,
                data=self.values,
            )

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> typing.Self:
        """Load event times previously stored with ``to_hdf()``."""
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='r') as file:
            if key not in file:
                raise KeyError(f'HDF5 key {key!r} does not exist')

            node = file[key]
            if not isinstance(node, h5py.Dataset):
                raise TypeError(f'Events payload {key!r} must be an HDF5 dataset')

            values = np.asarray(node[...])

        return cls(values)


class Events(nocte.core.hdf.HDFCollection[float]):
    """
    Indexed collection of point-like temporal occurrences.

    Each item has exactly one structural time, stored independently from metadata
    as an immutable finite float in milliseconds. Event identity is defined by
    the collection index rather than by timestamp value, so duplicate times are
    valid and represent distinct events.

    Public item-aligned values are returned as pandas objects so event identity
    is retained. Numerical operations use the private NumPy payload directly.
    """

    def __init__(
        self,
        data: _EventsData,
        meta: pd.DataFrame,
    ):
        self._data = data
        self.meta = meta.copy()

        if self.meta.index.name is None:
            self.meta.index = self.meta.index.rename('event_id')

        self._validate_meta(len(self._data))

    # ------------------------------------------------------------------
    # core collection and access

    @staticmethod
    def _default_meta(n_items: int) -> pd.DataFrame:
        return pd.DataFrame(
            index=pd.RangeIndex(n_items, name='event_id'),
        )

    def _sel_pos(self, positions: np.ndarray) -> typing.Self:
        return self.__class__(
            self._data.sel_pos(positions),
            self.meta.iloc[positions],
        )

    def _get_pos(self, position: int) -> float:
        return self._data.get_pos(position)

    def copy(self) -> typing.Self:
        """Return a collection sharing immutable times and copying metadata."""
        return self.__class__(self._data, self.meta)

    @property
    def time(self) -> pd.Series:
        """Event times indexed by stable event identity."""
        return pd.Series(
            self._time.copy(),
            index=self.index,
            name='time',
        )

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def from_times(
        cls,
        times: TimesLike,
        *,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        """
        Build events from scalar or one-dimensional times in milliseconds.

        A pandas Series contributes its index as event identity when ``meta`` is
        omitted. If both are supplied, their indices must match exactly.
        """
        if isinstance(times, pd.Series):
            values = times.to_numpy(dtype=float)

            if meta is None:
                meta = pd.DataFrame(index=times.index.copy())
            elif not meta.index.equals(times.index):
                raise ValueError('meta index must match times index')

            data = _EventsData(values)
        else:
            data = _EventsData.from_times(times)

            if meta is None:
                meta = cls._default_meta(len(data))

        return cls(data, meta)

    # ------------------------------------------------------------------
    # ordering and basic temporal transformations

    def sort_time(self, *, ascending: bool = True) -> typing.Self:
        """Return events ordered chronologically while preserving identities."""
        values = self._time if ascending else -self._time
        positions = np.argsort(values, kind='stable')
        return self._sel_pos(positions)

    def shift(self, by: EventValue) -> typing.Self:
        """Shift event times by a scalar or one value per event."""
        offsets = self._broadcast_value(by, name='by')
        return self.__class__(
            _EventsData(self._time + offsets),
            self.meta,
        )

    def round(
        self,
        decimals: int = 0,
        *,
        scale: nocte.core.time.TimeScale | float = 'milliseconds',
    ) -> typing.Self:
        """Round event times to a temporal scale."""
        scale_ms = nocte.core.time.scale_to_ms(scale)
        values = np.round(self._time / scale_ms, decimals=decimals) * scale_ms
        return self.__class__(
            _EventsData(values),
            self.meta,
        )

    # ------------------------------------------------------------------
    # temporal restriction and point-process summaries

    def contained_in(self, win: nocte.core.windows.Win) -> pd.Series:
        """Return an event-indexed mask for the half-open window ``win``."""
        start = win.time_at('start')
        stop = win.time_at('stop')
        mask = (start <= self._time) & (self._time < stop)
        return pd.Series(mask, index=self.index, name='contained_in')

    def crop(self, win: nocte.core.windows.Win) -> typing.Self:
        """Keep events inside the half-open window ``win``."""
        mask = self.contained_in(win).to_numpy()
        return self._sel_pos(np.flatnonzero(mask))

    def intervals(self) -> pd.Series:
        """
        Return time since the previous chronological event.

        Results are aligned back to event identity. The earliest event has NaN;
        simultaneous events have an interval of zero. Duplicate timestamps keep
        their stable current ordering.
        """
        result = np.full(len(self), np.nan, dtype=float)

        if len(self) >= 2:
            order = np.argsort(self._time, kind='stable')
            result[order[1:]] = np.diff(self._time[order])

        return pd.Series(
            result,
            index=self.index,
            name='interval',
        )

    def count_bins(self, bins: TimeArrayLike) -> pd.Series:
        """
        Count events in half-open bins ``[left, right)``.

        Events outside the supplied bin range are ignored. The final right edge
        remains exclusive so temporal bin semantics are uniform across bins.
        """
        edges = nocte.core._point_process.as_bin_edges(bins)
        values = nocte.core._point_process.count_bins_many(
            self._point_processes(),
            edges,
        )[0]

        return pd.Series(
            values,
            index=pd.IntervalIndex.from_breaks(
                edges,
                closed='left',
                name='time',
            ),
            name='count',
        )

    def count_rolling(
        self,
        window: float,
        *,
        step: float,
        within: nocte.core.windows.Win,
    ) -> pd.Series:
        """Count events in centered, half-open sliding windows."""
        window = self._positive_float(window, name='window')
        step = self._positive_float(step, name='step')

        sample_times = nocte.core._point_process.sample_centers(
            within.time_at('start'),
            within.time_at('stop'),
            step,
            margin=window * 0.5,
        )
        values = nocte.core._point_process.count_rolling_many(
            self._point_processes(),
            sample_times,
            window,
        )[0]

        return pd.Series(
            values,
            index=pd.Index(sample_times, name='time'),
            name='count',
        )

    def rate_gaussian(
        self,
        sigma: float,
        *,
        step: float,
        within: nocte.core.windows.Win,
        width: float = 5.0,
    ) -> pd.Series:
        """
        Estimate instantaneous event rate with a truncated Gaussian kernel.

        ``sigma``, ``step``, and ``within`` use milliseconds. The returned rate
        is expressed in events per second (Hz). ``width`` is the kernel
        half-width in multiples of ``sigma``. Evaluation centers are restricted
        so the complete truncated kernel lies inside ``within``.
        """
        sigma = self._positive_float(sigma, name='sigma')
        step = self._positive_float(step, name='step')
        width = self._positive_float(width, name='width')

        sample_times = nocte.core._point_process.sample_centers(
            within.time_at('start'),
            within.time_at('stop'),
            step,
            margin=sigma * width,
        )
        values = nocte.core._point_process.gaussian_rate_many(
            self._point_processes(),
            sample_times,
            sigma,
            width,
        )[0]
        values *= nocte.core.time.ms(seconds=1)

        return pd.Series(
            values,
            index=pd.Index(sample_times, name='time'),
            name='rate',
        )

    # ------------------------------------------------------------------
    # serialization

    def _to_hdf_data(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        self._data.to_hdf(
            path,
            key=f'{key}/data',
        )

    @classmethod
    def _from_hdf_data(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        meta: pd.DataFrame,
    ) -> typing.Self:
        data = _EventsData.from_hdf(
            path,
            key=f'{key}/data',
        )
        return cls(data, meta)

    # ------------------------------------------------------------------
    # representation

    def to_frame(self) -> pd.DataFrame:
        """Return structural time and metadata as one inspection DataFrame."""
        geometry = self.time.to_frame()
        collisions = geometry.columns.intersection(self.meta.columns)

        if len(collisions):
            warnings.warn(
                'to_frame() returns duplicate columns for structural names also '
                f'present in meta: {collisions.tolist()}',
                UserWarning,
                stacklevel=2,
            )

        return pd.concat([geometry, self.meta], axis=1)

    def _repr_html_(self) -> str:
        return self.to_frame()._repr_html_()  # type: ignore

    # ------------------------------------------------------------------
    # private numerical helpers

    @property
    def _time(self) -> np.ndarray:
        return self._data.values

    def _point_processes(self) -> tuple[np.ndarray]:
        """
        Return this event collection as the singleton point-process case.

        Point-process numerical helpers require sorted timestamps. Events do not
        require chronological collection order, so sort only when necessary.
        """
        times = self._time

        if len(times) >= 2 and np.any(times[1:] < times[:-1]):
            times = np.sort(times, kind='stable')

        return (times,)

    @staticmethod
    def _positive_float(value: float, *, name: str) -> float:
        value = float(value)

        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')

        return value

    def _broadcast_value(
        self,
        values: EventValue,
        *,
        name: str,
    ) -> np.ndarray:
        if isinstance(values, pd.Series):
            values = self._align_series(values, name)
            result = values.to_numpy(dtype=float)
        else:
            result = np.asarray(values, dtype=float)

            if result.ndim == 0:
                result = np.full(len(self), float(result))
            elif result.ndim != 1 or len(result) != len(self):
                raise ValueError(f'{name} must be scalar or have one value per event')

        if not np.isfinite(result).all():
            raise ValueError(f'{name} must be finite')

        return result
