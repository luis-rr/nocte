"""Temporal window definitions and collections."""

from __future__ import annotations

import collections.abc
import dataclasses
import logging
import pathlib
import typing
import warnings

import h5py
import numpy as np
import pandas as pd

import nocte.core.hdf
from nocte.core import time
from nocte.core.hdf import HDFCollection

logger = logging.getLogger(__name__)

WinPoint = typing.Literal['start', 'mid', 'ref', 'stop']
WinPosition = typing.Literal['start', 'mid', 'stop']
SnapMode = typing.Literal['nearest', 'loose', 'tight']

WinValue = float | collections.abc.Sequence[float] | np.ndarray | pd.Series
TimeArrayLike = float | collections.abc.Sequence[float] | np.ndarray
TimesLike = TimeArrayLike | pd.Index | pd.Series

MergeTake = typing.Literal['first', 'last']
SplitAlign = typing.Literal['left', 'right'] | float


@dataclasses.dataclass(frozen=True, slots=True)
class Win:
    """
    A temporal window expressed in milliseconds.

    `start` and `stop` are offsets relative to `ref`, while `ref` locates
    the window in its enclosing temporal coordinate system. The represented
    interval is always half-open: `[ref + start, ref + stop)`.

    All values are stored as finite float milliseconds. Empty windows are
    valid, contain no time, and have zero duration.
    """

    start: float
    stop: float
    ref: float = dataclasses.field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        start = float(self.start)
        stop = float(self.stop)
        ref = float(self.ref)

        if not np.all(np.isfinite([start, stop, ref])):
            raise ValueError('Window times must be finite')

        if stop < start:
            raise ValueError(
                f'Window stop ({stop}) must be greater than or equal to start ({start})'
            )

        object.__setattr__(self, 'start', start)
        object.__setattr__(self, 'stop', stop)
        object.__setattr__(self, 'ref', ref)

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def in_units(
        cls,
        start: float,
        stop: float,
        unit: time.TimeScale,
    ) -> typing.Self:
        """Build a window whose start and stop are expressed in `unit`."""
        scale = time.scale_to_ms(unit)
        return cls(float(start) * scale, float(stop) * scale)

    @classmethod
    def from_center(
        cls,
        center: float,
        duration: float,
        *,
        ref: float = 0.0,
    ) -> typing.Self:
        """Build a window of `duration` centered on `center`."""
        center = float(center)
        duration = float(duration)
        ref = float(ref)

        if not np.all(np.isfinite([center, duration, ref])):
            raise ValueError('Center, duration, and reference must be finite')

        if duration < 0:
            raise ValueError('Duration must be non-negative')

        center_offset = center - ref
        half_duration = duration * 0.5
        return cls(
            center_offset - half_duration,
            center_offset + half_duration,
            ref=ref,
        )

    # ------------------------------------------------------------------
    # geometry

    @property
    def length(self) -> float:
        """Window duration in milliseconds."""
        return self.stop - self.start

    @property
    def mid(self) -> float:
        """Window midpoint in the enclosing temporal coordinate."""
        return self.time_at('mid')

    def time_at(self, q: float | WinPoint) -> float:
        """
        Return a time point in the enclosing temporal coordinate.

        Numeric values are fractional positions through the interval, where
        0 is the start and 1 is the stop. `'ref'` returns the reference itself.
        """
        if isinstance(q, str):
            if q == 'start':
                return self.ref + self.start
            if q == 'mid':
                q = 0.5
            elif q == 'ref':
                return self.ref
            elif q == 'stop':
                return self.ref + self.stop
            else:
                raise ValueError(f'Unknown window point: {q!r}')

        q = float(q)
        if not np.isfinite(q) or not 0.0 <= q <= 1.0:
            raise ValueError('Window position must be between 0 and 1')

        return self.ref + self.start + self.length * q

    # ------------------------------------------------------------------
    # evaluation

    def is_empty(self) -> bool:
        """Return whether the window has zero duration."""
        return self.start == self.stop

    def contains(self, t: float) -> bool:
        """Return whether one time point falls within this half-open window."""
        t = float(t)
        if not np.isfinite(t):
            raise ValueError('Time must be finite')
        return self.time_at('start') <= t < self.time_at('stop')

    def contains_many(self, times: TimesLike) -> pd.Series:
        """Return an index-preserving mask for multiple time points."""
        series = _as_time_series(times)
        values = series.to_numpy(dtype=float)
        mask = (self.time_at('start') <= values) & (values < self.time_at('stop'))
        return pd.Series(mask, index=series.index, name='contains')

    def __contains__(self, t: float) -> bool:
        return self.contains(t)

    def contained_in(self, other: typing.Self) -> bool:
        """Return whether this window is fully contained in `other`."""
        return other.time_at('start') <= self.time_at('start') and self.time_at(
            'stop'
        ) <= other.time_at('stop')

    def overlaps(self, other: typing.Self) -> bool:
        """Return whether this window and `other` share positive-duration time."""
        start = max(self.time_at('start'), other.time_at('start'))
        stop = min(self.time_at('stop'), other.time_at('stop'))
        return start < stop

    # ------------------------------------------------------------------
    # relative construction

    def around(
        self,
        win: WinLike,
        *,
        q: float | WinPoint = 'mid',
    ) -> typing.Self:
        """Place `win` around a selected point of this window."""
        if not isinstance(win, Win):
            win = Win(*win)

        return self.__class__(
            win.start,
            win.stop,
            ref=self.time_at(q) + win.ref,
        )

    def centered(
        self,
        duration: float,
        *,
        q: float | WinPoint = 'mid',
    ) -> typing.Self:
        """Build a window of `duration` centered on a selected point."""
        return self.around(
            Win.from_center(0.0, duration),
            q=q,
        )

    def before(
        self,
        duration: float,
        *,
        offset: float = 0.0,
        q: float | WinPoint = 'start',
    ) -> typing.Self:
        """Build a window before a selected point."""
        duration = float(duration)
        offset = float(offset)

        if not np.all(np.isfinite([duration, offset])):
            raise ValueError('Duration and offset must be finite')
        if duration < 0:
            raise ValueError('Duration must be non-negative')

        return self.around(
            Win(-duration - offset, -offset),
            q=q,
        )

    def after(
        self,
        duration: float,
        *,
        offset: float = 0.0,
        q: float | WinPoint = 'stop',
    ) -> typing.Self:
        """Build a window after a selected point."""
        duration = float(duration)
        offset = float(offset)

        if not np.all(np.isfinite([duration, offset])):
            raise ValueError('Duration and offset must be finite')
        if duration < 0:
            raise ValueError('Duration must be non-negative')

        return self.around(
            Win(offset, duration + offset),
            q=q,
        )

    # ------------------------------------------------------------------
    # geometry transformations

    def change(
        self,
        pre: float = 0.0,
        post: float = 0.0,
    ) -> typing.Self:
        """Add `pre` to start and `post` to stop, preserving the reference."""
        return dataclasses.replace(
            self,
            start=self.start + float(pre),
            stop=self.stop + float(post),
        )

    def shrink(self, duration: float = 0.0) -> typing.Self:
        """Shrink both sides by the same non-negative duration."""
        duration = float(duration)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError('Duration must be finite and non-negative')
        return self.change(duration, -duration)

    def expand(self, duration: float = 0.0) -> typing.Self:
        """Expand both sides by the same non-negative duration."""
        duration = float(duration)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError('Duration must be finite and non-negative')
        return self.change(-duration, duration)

    def shift(self, by: float = 0.0) -> typing.Self:
        """Move the interval in time while preserving relative geometry."""
        return dataclasses.replace(self, ref=self.ref + float(by))

    def reanchor(self, q: float | WinPoint) -> typing.Self:
        """Change the reference while preserving the realized interval."""
        new_ref = self.time_at(q)
        delta = new_ref - self.ref
        return self.__class__(
            self.start - delta,
            self.stop - delta,
            ref=new_ref,
        )

    def crop(self, other: typing.Self) -> typing.Self:
        """
        Crop this window to `other`, preserving this window's reference.

        Disjoint windows produce an empty window at the nearest boundary of
        `other`.
        """
        self_start = self.time_at('start')
        self_stop = self.time_at('stop')
        other_start = other.time_at('start')
        other_stop = other.time_at('stop')

        start = min(max(self_start, other_start), other_stop)
        stop = max(min(self_stop, other_stop), other_start)

        return self.__class__(
            start - self.ref,
            stop - self.ref,
            ref=self.ref,
        )

    def shift_within(self, other: typing.Self) -> typing.Self:
        """Shift this window by the minimum amount needed to fit within `other`."""
        if self.length > other.length:
            raise ValueError(f'Window {self} cannot fit within {other}')

        self_start = self.time_at('start')
        self_stop = self.time_at('stop')
        other_start = other.time_at('start')
        other_stop = other.time_at('stop')

        if self_start < other_start:
            return self.shift(other_start - self_start)
        if self_stop > other_stop:
            return self.shift(other_stop - self_stop)
        return self

    def cap(
        self,
        max_duration: float,
        *,
        q: float | WinPosition = 'mid',
    ) -> typing.Self:
        """
        Limit duration while preserving a relative position through the window.

        `q='start'`, `'mid'`, and `'stop'` preserve the corresponding position.
        Numeric values preserve the same fractional position in the original and
        capped windows.
        """
        max_duration = float(max_duration)
        if not np.isfinite(max_duration) or max_duration < 0:
            raise ValueError('Maximum duration must be finite and non-negative')

        position = _position_fraction(q)

        if self.length <= max_duration:
            return self

        excess = self.length - max_duration
        return self.__class__(
            self.start + excess * position,
            self.stop - excess * (1.0 - position),
            ref=self.ref,
        )

    # ------------------------------------------------------------------
    # time generation

    def arange(self, step: float) -> np.ndarray:
        """Return regularly spaced times across the half-open window."""
        step = float(step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Step must be finite and positive')

        return np.arange(
            self.time_at('start'),
            self.time_at('stop'),
            step,
        )

    # ------------------------------------------------------------------
    # quantization

    def round(
        self,
        decimals: int = 0,
        *,
        start: bool = True,
        stop: bool = True,
        scale: time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round relative start and/or stop offsets."""
        return dataclasses.replace(
            self,
            start=(
                time.ms_round(
                    self.start,
                    scale=scale,
                    decimals=decimals,
                )
                if start
                else self.start
            ),
            stop=(
                time.ms_round(
                    self.stop,
                    scale=scale,
                    decimals=decimals,
                )
                if stop
                else self.stop
            ),
        )

    def floor(
        self,
        *,
        start: bool = True,
        stop: bool = True,
        scale: time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round relative start and/or stop offsets down."""
        return dataclasses.replace(
            self,
            start=(time.ms_floor(self.start, scale=scale) if start else self.start),
            stop=(time.ms_floor(self.stop, scale=scale) if stop else self.stop),
        )

    def ceil(
        self,
        *,
        start: bool = True,
        stop: bool = True,
        scale: time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round relative start and/or stop offsets up."""
        return dataclasses.replace(
            self,
            start=(time.ms_ceil(self.start, scale=scale) if start else self.start),
            stop=(time.ms_ceil(self.stop, scale=scale) if stop else self.stop),
        )

    def snap(
        self,
        mode: SnapMode = 'nearest',
        *,
        scale: time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Snap both boundaries to the nearest, outer, or inner grid points."""
        if mode == 'nearest':
            return self.round(scale=scale)
        if mode == 'loose':
            return self.floor(start=True, stop=False, scale=scale).ceil(
                start=False,
                stop=True,
                scale=scale,
            )
        if mode == 'tight':
            return self.ceil(start=True, stop=False, scale=scale).floor(
                start=False,
                stop=True,
                scale=scale,
            )
        raise ValueError(f'Unknown snap mode: {mode!r}')


WinLike = Win | tuple[float, float]


class _WindowsData:
    """Internal immutable storage for window geometry."""

    _START = 0
    _STOP = 1
    _REF = 2

    def __init__(self, values: np.ndarray):
        values = np.ascontiguousarray(values, dtype=float)

        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError('Window geometry must be a 2D array with 3 columns')
        if not np.isfinite(values).all():
            raise ValueError('Window geometry must be finite')
        if np.any(values[:, self._STOP] < values[:, self._START]):
            raise ValueError('Window stop must be greater than or equal to start')

        values.flags.writeable = False
        self.values = values

    @classmethod
    def from_arrays(
        cls,
        start: TimeArrayLike,
        stop: TimeArrayLike,
        ref: TimeArrayLike,
    ) -> typing.Self:
        values = [np.asarray(value, dtype=float) for value in (start, stop, ref)]

        if any(value.ndim > 1 for value in values):
            raise ValueError('start, stop, and ref must be scalar or one-dimensional')

        lengths = {len(value) for value in values if value.ndim == 1}
        if len(lengths) > 1:
            raise ValueError('start, stop, and ref arrays must have the same length')

        n = lengths.pop() if lengths else 1
        start_, stop_, ref_ = [
            np.full(n, value.item(), dtype=float) if value.ndim == 0 else value
            for value in values
        ]

        return cls(np.column_stack([start_, stop_, ref_]))

    @property
    def start(self) -> np.ndarray:
        return self.values[:, self._START]

    @property
    def stop(self) -> np.ndarray:
        return self.values[:, self._STOP]

    @property
    def ref(self) -> np.ndarray:
        return self.values[:, self._REF]

    def __len__(self) -> int:
        return len(self.values)

    def sel_pos(self, positions: np.ndarray) -> typing.Self:
        return self.__class__(self.values[positions])

    def get_pos(self, position: int) -> Win:
        return Win(
            start=self.start[position],
            stop=self.stop[position],
            ref=self.ref[position],
        )

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        """
        Store the window geometry payload.

        The payload is a two-dimensional HDF5 dataset with columns
        ``(start, stop, ref)``.

        The target key must not already exist.
        """
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
        """
        Load window geometry previously stored with to_hdf().
        """
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='r') as file:
            if key not in file:
                raise KeyError(f'HDF5 key {key!r} does not exist')

            node = file[key]

            if not isinstance(node, h5py.Dataset):
                raise TypeError(f'Windows payload {key!r} must be an HDF5 dataset')

            values = np.asarray(node[...])

        return cls(values)


class Windows(HDFCollection[Win]):
    """
    Indexed collection of temporal windows.

    Geometry is stored independently from metadata as immutable float
    milliseconds `(start, stop, ref)`. `start` and `stop` are offsets relative
    to `ref`, and every realized interval is half-open.

    Empty windows are valid items. They contain no time and contribute zero
    temporal coverage. Creating a collection containing empties emits a warning.

    Public item-aligned values are returned as pandas Series so item identity is
    retained. Numerical kernels operate on private NumPy arrays.
    """

    def __init__(
        self,
        data: _WindowsData,
        meta: pd.DataFrame,
    ):
        self._data = data
        self.meta = meta.copy()

        self._validate_meta(len(self._data))

        n_empty = int(np.count_nonzero(self._is_empty()))
        if n_empty:
            logger.warning('%d/%d windows are empty', n_empty, len(self))

    # ------------------------------------------------------------------
    # core collection and access

    def _sel_pos(self, positions: np.ndarray) -> typing.Self:
        return self.__class__(
            self._data.sel_pos(positions),
            self.meta.iloc[positions],
        )

    def _get_pos(self, position: int) -> Win:
        return self._data.get_pos(position)

    def copy(self) -> typing.Self:
        """Return a collection sharing immutable geometry and copying metadata."""
        return self.__class__(self._data, self.meta)

    @property
    def start(self) -> pd.Series:
        """Start offsets indexed by item identity."""
        return pd.Series(self._start.copy(), index=self.index, name='start')

    @property
    def stop(self) -> pd.Series:
        """Stop offsets indexed by item identity."""
        return pd.Series(self._stop.copy(), index=self.index, name='stop')

    @property
    def ref(self) -> pd.Series:
        """References indexed by item identity."""
        return pd.Series(self._ref.copy(), index=self.index, name='ref')

    @property
    def lengths(self) -> pd.Series:
        """Window durations indexed by item identity."""
        return pd.Series(self._lengths, index=self.index, name='length')

    @property
    def mid(self) -> pd.Series:
        """Window midpoints indexed by item identity."""
        return pd.Series(self._time_at('mid'), index=self.index, name='mid')

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def from_arrays(
        cls,
        start: TimeArrayLike,
        stop: TimeArrayLike,
        ref: TimeArrayLike = 0.0,
        *,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        """
        Build windows from scalar or one-dimensional geometry.

        Scalars are broadcast. One-dimensional inputs are itemwise values and
        must have equal length.
        """
        data = _WindowsData.from_arrays(start=start, stop=stop, ref=ref)
        return cls(
            data,
            meta=cls._default_meta(len(data)) if meta is None else meta,
        )

    @classmethod
    def build_around(
        cls,
        marks: TimeArrayLike | pd.Series,
        win: WinLike = (0.0, 0.0),
        *,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        """Instantiate the same relative window around multiple reference times."""
        if not isinstance(win, Win):
            win = Win(*win)

        if isinstance(marks, pd.Series):
            mark_values = marks.to_numpy(dtype=float)
            mark_index = marks.index

            if meta is None:
                meta = pd.DataFrame(index=mark_index.copy())
            elif not meta.index.equals(mark_index):
                raise ValueError('meta index must match marks index')
        else:
            mark_values = np.asarray(marks, dtype=float)
            if mark_values.ndim == 0:
                mark_values = mark_values.reshape(1)
            elif mark_values.ndim != 1:
                raise ValueError('marks must be scalar or one-dimensional')

        if not np.isfinite(mark_values).all():
            raise ValueError('marks must be finite')

        return cls.from_arrays(
            start=win.start,
            stop=win.stop,
            ref=mark_values + win.ref,
            meta=meta,
        )

    @classmethod
    def build_centered(
        cls,
        marks: TimeArrayLike | pd.Series,
        duration: float,
        *,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        """Build equal-duration windows centered on multiple reference times."""
        return cls.build_around(
            marks,
            Win.from_center(0.0, duration),
            meta=meta,
        )

    @classmethod
    def build_between(
        cls,
        times: TimeArrayLike | pd.Series,
    ) -> typing.Self:
        """
        Build windows between successive sorted time points.

        Each result uses its left edge as reference. For a Series input, source
        identities of both bounding markers are retained in metadata.
        """
        source_ids = None

        if isinstance(times, pd.Series):
            values = times.to_numpy(dtype=float)
            source_ids = times.index.to_numpy()
            source_name = times.index.name or 'source_id'
        else:
            values = np.asarray(times, dtype=float)
            source_name = None

        if values.ndim == 0:
            values = values.reshape(1)
        elif values.ndim != 1:
            raise ValueError('times must be scalar or one-dimensional')
        if not np.isfinite(values).all():
            raise ValueError('times must be finite')

        order = np.argsort(values, kind='stable')
        values = values[order]
        if source_ids is not None:
            source_ids = source_ids[order]

        if len(values) < 2:
            return cls.from_arrays([], [])

        refs = values[:-1]
        stops = np.diff(values)
        meta = None

        if source_ids is not None:
            meta = pd.DataFrame(
                {
                    f'start_{source_name}': source_ids[:-1],
                    f'stop_{source_name}': source_ids[1:],
                }
            )

        return cls.from_arrays(
            start=0.0,
            stop=stops,
            ref=refs,
            meta=meta,
        )

    @classmethod
    def from_dict(
        cls,
        wins: collections.abc.Mapping[typing.Any, WinLike],
        *,
        name: str = 'cat',
    ) -> typing.Self:
        """Build windows from a mapping of metadata values to definitions."""
        labels: list[typing.Any] = []
        start: list[float] = []
        stop: list[float] = []
        ref: list[float] = []

        for label, win in wins.items():
            if not isinstance(win, Win):
                win = Win(*win)
            labels.append(label)
            start.append(win.start)
            stop.append(win.stop)
            ref.append(win.ref)

        return cls.from_arrays(
            start,
            stop,
            ref,
            meta=pd.DataFrame({name: labels}),
        )

    @classmethod
    def from_contiguous_values(
        cls,
        values: pd.Series,
        *,
        step: float | None = None,
        name: str | None = None,
    ) -> typing.Self:
        """
        Build tight windows from contiguous runs in a regularly sampled Series.

        The Series index is interpreted as sample-center time in milliseconds;
        transitions lie halfway between samples. `step` is inferred when at
        least two samples are available and is required for a one-sample input.
        """
        if not isinstance(values, pd.Series):
            raise TypeError('values must be a pandas Series with a time index')

        column = (
            name
            if name is not None
            else values.name
            if isinstance(values.name, str)
            else 'cat'
        )

        if values.isna().any():
            raise ValueError('values must not contain missing entries')

        if len(values) == 0:
            return cls.from_arrays(
                [],
                [],
                [],
                meta=pd.DataFrame({column: pd.Series(dtype=values.dtype)}),
            )

        times = values.index.to_numpy(dtype=float)
        if not np.isfinite(times).all():
            raise ValueError('values index must contain finite times')

        if step is not None:
            step = float(step)
            if not np.isfinite(step) or step <= 0:
                raise ValueError('step must be finite and positive')

        if len(times) == 1:
            if step is None:
                raise ValueError('step is required for a one-sample Series')
        else:
            diffs = np.diff(times)
            if np.any(diffs <= 0):
                raise ValueError('values index must be strictly increasing')

            if step is None:
                step = float(diffs[0])

            assert step is not None
            tol = (
                32
                * np.finfo(float).eps
                * max(
                    1.0,
                    abs(step),
                    float(np.max(np.abs(times))),
                )
            )
            if not np.allclose(diffs, step, rtol=0.0, atol=tol):
                raise ValueError('values index must be regularly sampled')

        assert step is not None

        array = values.to_numpy()
        changes = np.empty(len(array), dtype=bool)
        changes[0] = True
        changes[1:] = array[1:] != array[:-1]

        run_start = np.flatnonzero(changes)
        run_stop = np.concatenate([run_start[1:], [len(array)]])
        realized_start = times[run_start] - step * 0.5
        realized_stop = times[run_stop - 1] + step * 0.5

        meta = values.iloc[run_start].rename(column).to_frame().reset_index(drop=True)

        return cls.from_arrays(
            start=0.0,
            stop=realized_stop - realized_start,
            ref=realized_start,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # geometry and evaluation

    def time_at(self, q: float | WinPoint) -> pd.Series:
        """Return one enclosing-coordinate time per window."""
        name = q if isinstance(q, str) else 'time'
        return pd.Series(self._time_at(q), index=self.index, name=name)

    def contains(self, t: float) -> pd.Series:
        """Return an item-indexed mask for windows containing one time point."""
        t = float(t)
        if not np.isfinite(t):
            raise ValueError('Time must be finite')
        start = self._time_at('start')
        stop = self._time_at('stop')
        return pd.Series(
            (start <= t) & (t < stop),
            index=self.index,
            name='contains',
        )

    def contained_in(self, win: Win) -> pd.Series:
        """Return an item-indexed mask for windows fully contained in `win`."""
        win_start = win.time_at('start')
        win_stop = win.time_at('stop')
        mask = (win_start <= self._time_at('start')) & (
            self._time_at('stop') <= win_stop
        )
        return pd.Series(mask, index=self.index, name='contained_in')

    def overlaps(self, win: Win) -> pd.Series:
        """Return an item-indexed mask for windows overlapping `win`."""
        start = np.maximum(self._time_at('start'), win.time_at('start'))
        stop = np.minimum(self._time_at('stop'), win.time_at('stop'))
        return pd.Series(start < stop, index=self.index, name='overlaps')

    def is_empty(self) -> pd.Series:
        """Return an item-indexed mask for zero-duration windows."""
        return pd.Series(self._is_empty(), index=self.index, name='is_empty')

    # ------------------------------------------------------------------
    # per-window transformations

    def around(
        self,
        win: WinLike,
        *,
        q: float | WinPoint = 'mid',
    ) -> typing.Self:
        """Build a new window around one selected point from each current window."""
        marks = pd.Series(self._time_at(q), index=self.index)
        return self.__class__.build_around(marks, win, meta=self.meta)

    def centered(
        self,
        duration: float,
        *,
        q: float | WinPoint = 'mid',
    ) -> typing.Self:
        """Build equal-duration windows centered on one selected point each."""
        return self.around(Win.from_center(0.0, duration), q=q)

    def before(
        self,
        duration: float,
        *,
        offset: float = 0.0,
        q: float | WinPoint = 'start',
    ) -> typing.Self:
        """Build windows before one selected point from each current window."""
        duration = float(duration)
        offset = float(offset)
        if not np.all(np.isfinite([duration, offset])):
            raise ValueError('Duration and offset must be finite')
        if duration < 0:
            raise ValueError('Duration must be non-negative')
        return self.around(Win(-duration - offset, -offset), q=q)

    def after(
        self,
        duration: float,
        *,
        offset: float = 0.0,
        q: float | WinPoint = 'stop',
    ) -> typing.Self:
        """Build windows after one selected point from each current window."""
        duration = float(duration)
        offset = float(offset)
        if not np.all(np.isfinite([duration, offset])):
            raise ValueError('Duration and offset must be finite')
        if duration < 0:
            raise ValueError('Duration must be non-negative')
        return self.around(Win(offset, duration + offset), q=q)

    def change(
        self,
        pre: WinValue = 0.0,
        post: WinValue = 0.0,
    ) -> typing.Self:
        pre_ = self._broadcast_value(pre, name='pre')
        post_ = self._broadcast_value(post, name='post')
        return self.__class__.from_arrays(
            self._start + pre_,
            self._stop + post_,
            self._ref,
            meta=self.meta,
        )

    def shrink(self, duration: WinValue = 0.0) -> typing.Self:
        """Shrink both sides by non-negative per-window durations."""
        duration_ = self._broadcast_value(duration, name='duration')
        if np.any(duration_ < 0):
            raise ValueError('Duration must be non-negative')
        return self.change(duration_, -duration_)

    def expand(self, duration: WinValue = 0.0) -> typing.Self:
        """Expand both sides by non-negative per-window durations."""
        duration_ = self._broadcast_value(duration, name='duration')
        if np.any(duration_ < 0):
            raise ValueError('Duration must be non-negative')
        return self.change(-duration_, duration_)

    def shift(self, by: WinValue = 0.0) -> typing.Self:
        """Shift windows in time while preserving relative geometry."""
        by_ = self._broadcast_value(by, name='shift')
        return self.__class__.from_arrays(
            self._start,
            self._stop,
            self._ref + by_,
            meta=self.meta,
        )

    def reanchor(self, q: float | WinPoint) -> typing.Self:
        """Change each reference while preserving each realized interval."""
        new_ref = self._time_at(q)
        delta = new_ref - self._ref
        return self.__class__.from_arrays(
            self._start - delta,
            self._stop - delta,
            new_ref,
            meta=self.meta,
        )

    def crop(self, win: Win) -> typing.Self:
        """Crop every window to `win`, preserving identity and reference."""
        start = self._time_at('start')
        stop = self._time_at('stop')
        win_start = win.time_at('start')
        win_stop = win.time_at('stop')

        cropped_start = np.minimum(np.maximum(start, win_start), win_stop)
        cropped_stop = np.maximum(np.minimum(stop, win_stop), win_start)

        return self.__class__.from_arrays(
            cropped_start - self._ref,
            cropped_stop - self._ref,
            self._ref,
            meta=self.meta,
        )

    def drop_empty(self) -> typing.Self:
        """Drop zero-duration windows."""
        return self.sel_mask(~self.is_empty())

    # ------------------------------------------------------------------
    # collection geometry

    def are_uniform(self, atol: float = 1e-8) -> bool:
        """Return whether all windows share the same relative geometry."""
        atol = float(atol)
        if not np.isfinite(atol) or atol < 0:
            raise ValueError('atol must be finite and non-negative')
        if len(self) <= 1:
            return True

        return bool(
            np.allclose(self._start, self._start[0], rtol=0.0, atol=atol)
            and np.allclose(self._stop, self._stop[0], rtol=0.0, atol=atol)
        )

    def are_exclusive(self) -> bool:
        """Return whether positive-duration coverage does not overlap."""
        start, stop = self._sorted_coverage_bounds()
        if len(start) <= 1:
            return True
        return bool(np.all(start[1:] >= np.maximum.accumulate(stop[:-1])))

    def are_tight(self) -> bool:
        """Return whether positive-duration coverage has no temporal gaps."""
        start, stop = self._sorted_coverage_bounds()
        if len(start) <= 1:
            return True
        return bool(np.all(start[1:] <= np.maximum.accumulate(stop[:-1])))

    def bounding_win(self) -> Win:
        """Return the minimum window spanning all positive-duration coverage."""
        start, stop = self._sorted_coverage_bounds()
        if len(start) == 0:
            raise ValueError('cannot get a bounding window without non-empty coverage')

        first = float(start[0])
        last = float(stop.max())
        return Win(0.0, last - first, ref=first)

    def sort_time(self) -> typing.Self:
        """Sort windows by realized start, then stop time."""
        return self._sel_pos(self._time_order())

    def edges(self) -> np.ndarray:
        """Return sorted unique realized edges, including empty items."""
        start, stop = self._sorted_realized_bounds()
        return np.unique(np.concatenate([start, stop]))

    def breaks(self) -> np.ndarray:
        """Return ordered boundaries of tight, exclusive positive-duration coverage."""
        if not self.are_exclusive() or not self.are_tight():
            raise ValueError('breaks require windows to be both tight and exclusive')

        start, stop = self._sorted_coverage_bounds()
        if len(start) == 0:
            return np.empty(0, dtype=float)
        return np.concatenate([start, stop[-1:]])

    # ------------------------------------------------------------------
    # structural transformations

    def merge_overlap(
        self,
        *,
        by: str | None = None,
        take: MergeTake = 'first',
    ) -> typing.Self:
        """Merge overlapping chronological neighbors, optionally constrained by `by`."""
        return self._merge_connected(touching=False, take=take, by=by)

    def merge_tight(
        self,
        *,
        by: str | None = None,
        take: MergeTake = 'first',
    ) -> typing.Self:
        """Merge overlapping or touching neighbors, optionally constrained by `by`."""
        return self._merge_connected(touching=True, take=take, by=by)

    def invert(self, within: Win) -> typing.Self:
        """Return all uncovered positive-duration regions inside `within`."""
        outer_start = within.time_at('start')
        outer_stop = within.time_at('stop')

        if outer_start == outer_stop:
            return self.__class__.from_arrays([], [], [])

        start = self._time_at('start')
        stop = self._time_at('stop')
        valid = (~self._is_empty()) & (start < outer_stop) & (outer_start < stop)
        start = np.maximum(start[valid], outer_start)
        stop = np.minimum(stop[valid], outer_stop)

        if len(start) == 0:
            return self.__class__.from_arrays(
                0.0, outer_stop - outer_start, outer_start
            )

        order = np.lexsort((stop, start))
        start = start[order]
        stop = stop[order]

        gap_start: list[float] = []
        gap_stop: list[float] = []
        cursor = outer_start

        for this_start, this_stop in zip(start, stop, strict=True):
            if this_start > cursor:
                gap_start.append(float(cursor))
                gap_stop.append(float(this_start))
            cursor = max(cursor, this_stop)
            if cursor >= outer_stop:
                break

        if cursor < outer_stop:
            gap_start.append(float(cursor))
            gap_stop.append(float(outer_stop))

        gap_start_ = np.asarray(gap_start, dtype=float)
        gap_stop_ = np.asarray(gap_stop, dtype=float)

        return self.__class__.from_arrays(
            start=np.zeros(len(gap_start_)),
            stop=gap_stop_ - gap_start_,
            ref=gap_start_,
        )

    def split(
        self,
        length: float,
        *,
        align: SplitAlign = 'left',
    ) -> typing.Self:
        """
        Split each window into complete equal-length fragments.

        `align` controls where any remainder is left. Result items receive new
        identities and record `source_win_id` and `fragment_idx` provenance.
        """
        length = float(length)
        if not np.isfinite(length) or length <= 0:
            raise ValueError('Split length must be finite and positive')

        if isinstance(align, str):
            try:
                align_value = {'left': 0.0, 'right': 1.0}[align]
            except KeyError:
                raise ValueError(f'Unknown alignment: {align!r}') from None
        else:
            align_value = float(align)
            if not np.isfinite(align_value) or not 0.0 <= align_value <= 1.0:
                raise ValueError('Numeric alignment must be between 0 and 1')

        for col in ('source_win_id', 'fragment_idx'):
            if col in self.meta.columns:
                raise ValueError(f'meta already contains {col!r}')

        starts: list[np.ndarray] = []
        stops: list[np.ndarray] = []
        refs: list[np.ndarray] = []
        source_positions: list[int] = []
        fragment_idcs: list[int] = []
        eps = np.finfo(float).eps

        for pos in range(len(self)):
            duration = self._lengths[pos]
            ratio = duration / length
            nearest = round(ratio)
            ratio_tol = 16 * eps * max(1.0, abs(ratio))
            n = (
                int(nearest)
                if abs(ratio - nearest) <= ratio_tol
                else int(np.floor(ratio))
            )

            if n == 0:
                continue

            remainder = duration - n * length
            duration_tol = 16 * eps * max(1.0, abs(duration), abs(n * length))
            if abs(remainder) <= duration_tol:
                remainder = 0.0

            offset = remainder * align_value
            fragment_start = self._start[pos] + offset + np.arange(n) * length

            starts.append(fragment_start)
            stops.append(fragment_start + length)
            refs.append(np.full(n, self._ref[pos]))
            source_positions.extend([pos] * n)
            fragment_idcs.extend(range(n))

        source_positions_ = np.asarray(source_positions, dtype=int)
        meta = self.meta.iloc[source_positions_].copy()
        meta.insert(0, 'source_win_id', self.index.to_numpy()[source_positions_])
        meta.insert(1, 'fragment_idx', fragment_idcs)
        meta.index = pd.RangeIndex(len(meta), name=self.index.name)

        if not starts:
            return self.__class__.from_arrays([], [], [], meta=meta)

        return self.__class__.from_arrays(
            start=np.concatenate(starts),
            stop=np.concatenate(stops),
            ref=np.concatenate(refs),
            meta=meta,
        )

    def defrag(self, start: float = 0.0) -> typing.Self:
        """Shift windows into a tight sequence in current item order."""
        start = float(start)
        if not np.isfinite(start):
            raise ValueError('Defragmentation start must be finite')
        if len(self) == 0:
            return self.copy()

        target_start = start + np.concatenate([[0.0], np.cumsum(self._lengths[:-1])])
        return self.shift(target_start - self._time_at('start'))

    # ------------------------------------------------------------------
    # temporal relationships

    def interval_to_prev(self, n: int = 1) -> pd.Series:
        """Return interval to the nth previous chronological window."""
        result = self._interval_to_prev(n)
        return pd.Series(result, index=self.index, name='interval_to_prev')

    def interval_to_next(self, n: int = 1) -> pd.Series:
        """Return interval to the nth next chronological window."""
        result = self._interval_to_next(n)
        return pd.Series(result, index=self.index, name='interval_to_next')

    def interval_to_closest(self) -> pd.Series:
        """Return the smaller interval to the immediate previous or next window."""
        result = np.minimum(self._interval_to_prev(1), self._interval_to_next(1))
        return pd.Series(result, index=self.index, name='interval_to_closest')

    def is_isolated(
        self,
        at_least: float | tuple[float, float],
    ) -> pd.Series:
        """Return whether exclusive windows meet requested pre/post separation."""
        if not self.are_exclusive():
            raise ValueError('is_isolated requires exclusive windows')

        if isinstance(at_least, tuple):
            pre, post = at_least
        else:
            pre = post = at_least

        pre = float(pre)
        post = float(post)
        if not np.all(np.isfinite([pre, post])) or pre < 0 or post < 0:
            raise ValueError('Isolation thresholds must be finite and non-negative')

        mask = (self._interval_to_prev(1) >= pre) & (self._interval_to_next(1) >= post)
        return pd.Series(mask, index=self.index, name='is_isolated')

    # ------------------------------------------------------------------
    # event matching and categorical operations

    def match_events(self, times: TimesLike) -> WindowMatches:
        """
        Return the sparse event-window relation as positional arrays.

        Match ordering is intentionally unspecified. Events outside all windows
        are omitted; overlapping windows may match the same event more than once.
        """
        return WindowMatches.from_times(self, _as_time_array(times))

    def classify_events(
        self,
        times: TimesLike,
        *,
        relative_to: float | WinPoint | None = None,
        cols: str | collections.abc.Iterable[str] = (),
    ) -> pd.DataFrame:
        """Return a tabular projection of event-window matches."""
        events = _as_time_series(times)
        values = events.to_numpy(dtype=float)

        if isinstance(cols, str):
            columns = [cols]
        else:
            columns = list(cols)

        missing = set(columns).difference(self.meta.columns)
        if missing:
            raise KeyError(f'Unknown metadata columns: {sorted(missing)}')

        collisions = {'item_id', 'delay'}.intersection(columns)
        if collisions:
            raise ValueError(f'cols conflict with output columns: {sorted(collisions)}')

        matches = WindowMatches.from_times(self, values)

        if len(matches.event_pos):
            order = np.lexsort((matches.win_pos, matches.event_pos))
            event_pos = matches.event_pos[order]
            win_pos = matches.win_pos[order]
        else:
            event_pos = matches.event_pos
            win_pos = matches.win_pos

        data: dict[str, typing.Any] = {
            'item_id': self.index.to_numpy()[win_pos],
        }

        if relative_to is not None:
            anchor = self._time_at(relative_to)
            data['delay'] = values[event_pos] - anchor[win_pos]

        for col in columns:
            data[col] = self.meta[col].to_numpy()[win_pos]

        return pd.DataFrame(data, index=events.index.take(event_pos))

    def generate(
        self,
        times: TimesLike,
        *,
        by: str,
        fill_value: typing.Any = np.nan,
    ) -> pd.Series:
        """Generate one metadata value per requested time for exclusive windows."""
        if by not in self.meta.columns:
            raise KeyError(f'Unknown metadata column: {by!r}')
        if not self.are_exclusive():
            raise ValueError('generate requires exclusive windows')

        events = _as_time_series(times)
        values = events.to_numpy(dtype=float)
        win_pos = _assign_events_exclusive(self, values)
        matched = win_pos >= 0

        generated = np.empty(len(values), dtype=object)
        generated[:] = fill_value
        generated[matched] = self.meta[by].to_numpy()[win_pos[matched]]

        return pd.Series(
            generated,
            index=events.index,
            name=by,
        ).infer_objects()

    def generate_contiguous(
        self,
        step: float,
        *,
        by: str,
        start: float | None = None,
        stop: float | None = None,
        fill_value: typing.Any = np.nan,
    ) -> pd.Series:
        """Generate a regularly sampled metadata Series indexed by sample time."""
        step = float(step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError('step must be finite and positive')

        coverage_start, coverage_stop = self._sorted_coverage_bounds()

        if len(coverage_start) == 0:
            if start is None and stop is None:
                empty_times = pd.Index([], dtype=float, name='time')
                return self.generate(empty_times, by=by, fill_value=fill_value)
            if start is None or stop is None:
                raise ValueError(
                    'start and stop are required when there is no temporal coverage'
                )

        if start is None:
            start = float(coverage_start[0] + step * 0.5)
        else:
            start = float(start)

        if stop is None:
            stop = float(coverage_stop.max())
        else:
            stop = float(stop)

        if not np.all(np.isfinite([start, stop])):
            raise ValueError('start and stop must be finite')
        if stop < start:
            raise ValueError('stop must be greater than or equal to start')

        times = np.arange(start, stop, step)
        return self.generate(
            pd.Index(times, name='time'),
            by=by,
            fill_value=fill_value,
        )

    def is_sandwiched(
        self,
        by: str,
        *,
        max_length: float | None = None,
        only: typing.Any = None,
    ) -> pd.Series:
        """Return items bracketed by equal, different categorical neighbors."""
        if by not in self.meta.columns:
            raise KeyError(f'Unknown metadata column: {by!r}')
        if self.meta[by].isna().any():
            raise ValueError(f'{by!r} contains missing values')
        if not self.are_exclusive() or not self.are_tight():
            raise ValueError('is_sandwiched requires tight, exclusive windows')

        order = self._time_order()
        order = order[~self._is_empty()[order]]
        categories = self.meta[by].to_numpy()[order]
        mask = np.zeros(len(self), dtype=bool)

        if len(order) >= 3:
            mask[order[1:-1]] = (categories[:-2] == categories[2:]) & (
                categories[1:-1] != categories[2:]
            )

        if max_length is not None:
            max_length = float(max_length)
            if not np.isfinite(max_length) or max_length < 0:
                raise ValueError('max_length must be finite and non-negative')
            mask &= self._lengths <= max_length

        if only is not None:
            if isinstance(only, collections.abc.Iterable) and not isinstance(
                only,
                (str, bytes),
            ):
                values = list(only)
            else:
                values = [only]
            mask &= self.meta[by].isin(values).to_numpy()

        return pd.Series(mask, index=self.index, name='is_sandwiched')

    def merge_sandwiched(
        self,
        by: str,
        *,
        max_length: float | None = None,
        only: typing.Any = None,
    ) -> typing.Self:
        """Relabel and merge windows bracketed by the same category until stable."""
        wins = self.sort_time()

        while True:
            mask = wins.is_sandwiched(by, max_length=max_length, only=only)
            if not mask.any():
                return wins

            order = wins._time_order()
            order = order[~wins._is_empty()[order]]
            selected = mask.to_numpy()[order]
            selected_in_order = np.flatnonzero(selected)
            source_pos = order[selected_in_order + 1]
            target_pos = order[selected_in_order]

            wins.meta.loc[wins.index[target_pos], by] = wins.meta[by].to_numpy()[
                source_pos
            ]
            wins = wins.merge_tight(by=by)

    # ------------------------------------------------------------------
    # representation

    def to_frame(self) -> pd.DataFrame:
        """Return relative geometry and metadata together as a new DataFrame."""
        geometry = pd.DataFrame(
            {
                'start': self._start,
                'stop': self._stop,
                'ref': self._ref,
            },
            index=self.index,
        )

        collisions = geometry.columns.intersection(self.meta.columns)
        if len(collisions):
            warnings.warn(
                'to_frame() returns duplicate columns for geometry names also present in meta: '
                f'{collisions.tolist()}',
                UserWarning,
                stacklevel=2,
            )

        return pd.concat([geometry, self.meta], axis=1)

    def _repr_html_(self) -> str:
        return self.to_frame()._repr_html_()  # type: ignore

    # ------------------------------------------------------------------
    # private numerical helpers

    @property
    def _start(self) -> np.ndarray:
        return self._data.start

    @property
    def _stop(self) -> np.ndarray:
        return self._data.stop

    @property
    def _ref(self) -> np.ndarray:
        return self._data.ref

    @property
    def _lengths(self) -> np.ndarray:
        return self._stop - self._start

    def _time_at(self, q: float | WinPoint) -> np.ndarray:
        if isinstance(q, str):
            if q == 'start':
                return self._ref + self._start
            if q == 'ref':
                return self._ref.copy()
            if q == 'stop':
                return self._ref + self._stop
            if q == 'mid':
                q = 0.5
            else:
                raise ValueError(f'Unknown window point: {q!r}')

        q = float(q)
        if not np.isfinite(q) or not 0.0 <= q <= 1.0:
            raise ValueError('Window position must be between 0 and 1')
        return self._ref + self._start + self._lengths * q

    def _is_empty(self) -> np.ndarray:
        return self._start == self._stop

    def _broadcast_value(
        self,
        values: WinValue,
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
                raise ValueError(f'{name} must be scalar or have one value per window')

        if not np.isfinite(result).all():
            raise ValueError(f'{name} must be finite')
        return result

    def _time_order(self) -> np.ndarray:
        """Return positions sorted by realized start, then stop."""
        return np.lexsort((self._time_at('stop'), self._time_at('start')))

    def _sorted_realized_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        start = self._time_at('start')
        stop = self._time_at('stop')
        order = self._time_order()
        return start[order], stop[order]

    def _sorted_coverage_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        start, stop = self._sorted_realized_bounds()
        nonempty = start < stop
        return start[nonempty], stop[nonempty]

    def _merge_connected(
        self,
        *,
        touching: bool,
        take: MergeTake,
        by: str | None,
    ) -> typing.Self:
        if take not in ('first', 'last'):
            raise ValueError(f'Unknown take policy: {take!r}')

        source_ids = (
            list(self.meta['source_win_ids'])
            if 'source_win_ids' in self.meta.columns
            else [(item_id,) for item_id in self.index]
        )
        if not all(isinstance(ids, tuple) for ids in source_ids):
            raise ValueError("'source_win_ids' must contain tuples")

        categories = None
        if by is not None:
            if by not in self.meta.columns:
                raise KeyError(f'Unknown metadata column: {by!r}')
            if self.meta[by].isna().any():
                raise ValueError(f'{by!r} contains missing values')
            categories = self.meta[by].to_numpy()

        start = self._time_at('start')
        stop = self._time_at('stop')
        nonempty = np.flatnonzero(start < stop)
        empty = np.flatnonzero(start == stop)
        order = nonempty[np.lexsort((stop[nonempty], start[nonempty]))]

        groups: list[list[int]] = []
        group_stop = -np.inf

        for pos_ in order:
            pos = int(pos_)
            same_group = bool(groups)

            if same_group:
                connected = (
                    start[pos] <= group_stop if touching else start[pos] < group_stop
                )
                same_category = categories is None or (
                    categories[pos] == categories[groups[-1][0]]
                )
                same_group = connected and bool(same_category)

            if same_group:
                groups[-1].append(pos)
                group_stop = max(group_stop, stop[pos])
            else:
                groups.append([pos])
                group_stop = stop[pos]

        groups.extend([[int(pos)] for pos in empty])
        groups.sort(key=lambda group: (start[group[0]], stop[group[0]]))

        sources = np.asarray(
            [group[0] if take == 'first' else group[-1] for group in groups],
            dtype=int,
        )
        refs = self._ref[sources]
        starts = np.asarray([start[group].min() for group in groups]) - refs
        stops = np.asarray([stop[group].max() for group in groups]) - refs

        meta = self.meta.iloc[sources].copy()
        meta['source_win_ids'] = [
            tuple(item_id for pos in group for item_id in source_ids[pos])
            for group in groups
        ]
        meta.index = pd.RangeIndex(len(meta), name=self.index.name)

        return self.from_arrays(starts, stops, refs, meta=meta)

    def _interval_to_prev(self, n: int) -> np.ndarray:
        n = _validate_neighbor_n(n)
        result = np.full(len(self), np.inf)
        if n >= len(self):
            return result

        start = self._time_at('start')
        stop = self._time_at('stop')
        order = self._time_order()
        current = order[n:]
        previous = order[:-n]
        result[current] = start[current] - stop[previous]
        return result

    def _interval_to_next(self, n: int) -> np.ndarray:
        n = _validate_neighbor_n(n)
        result = np.full(len(self), np.inf)
        if n >= len(self):
            return result

        start = self._time_at('start')
        stop = self._time_at('stop')
        order = self._time_order()
        current = order[:-n]
        following = order[n:]
        result[current] = start[following] - stop[current]
        return result

    def _to_hdf_data(
        self,
        path: str | pathlib.Path,
        *,
        key: str = 'traces',
    ) -> None:
        self._data.to_hdf(path, key=f'{key}/data')

    @classmethod
    def _from_hdf_data(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        meta: pd.DataFrame,
    ) -> typing.Self:

        data = _WindowsData.from_hdf(path, key=f'{key}/data')

        return cls(data=data, meta=meta)


def _position_fraction(q: float | WinPosition) -> float:
    if isinstance(q, str):
        try:
            return {'start': 0.0, 'mid': 0.5, 'stop': 1.0}[q]
        except KeyError:
            raise ValueError(f'Unknown window position: {q!r}') from None

    q = float(q)
    if not np.isfinite(q) or not 0.0 <= q <= 1.0:
        raise ValueError('Window position must be between 0 and 1')
    return q


def _validate_neighbor_n(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
        raise ValueError('n must be a positive integer')
    return int(n)


def _as_time_array(times: TimesLike) -> np.ndarray:
    if isinstance(times, (pd.Index, pd.Series)):
        values = times.to_numpy(dtype=float)
    else:
        values = np.asarray(times, dtype=float)

    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim != 1:
        raise ValueError('times must be scalar or one-dimensional')

    if not np.isfinite(values).all():
        raise ValueError('times must be finite')
    return values


def _as_time_series(times: TimesLike) -> pd.Series:
    values = _as_time_array(times)

    if isinstance(times, pd.Series):
        index = times.index.copy()
    elif isinstance(times, pd.Index):
        index = times.copy()
    else:
        index = pd.RangeIndex(len(values))

    return pd.Series(values, index=index, name='time')


@dataclasses.dataclass(frozen=True, slots=True)
class WindowMatches:
    """Aligned event and window positions for event-window matches."""

    event_pos: np.ndarray
    win_pos: np.ndarray

    @classmethod
    def build_empty(cls) -> WindowMatches:
        empty = np.empty(0, dtype=int)
        return cls(event_pos=empty, win_pos=empty)

    @classmethod
    def from_times(
        cls,
        windows: Windows,
        times: np.ndarray,
    ) -> WindowMatches:
        """Return every event-window match without imposing presentation order."""

        win_positions = np.flatnonzero(~windows._is_empty())

        if len(times) == 0 or len(win_positions) == 0:
            return WindowMatches.build_empty()

        time_order = np.argsort(times, kind='stable')
        sorted_times = times[time_order]
        start = windows._time_at('start')
        stop = windows._time_at('stop')

        event_positions: list[np.ndarray] = []
        matched_windows: list[np.ndarray] = []

        for win_pos_ in win_positions:
            win_pos = int(win_pos_)
            left = np.searchsorted(sorted_times, start[win_pos], side='left')
            right = np.searchsorted(sorted_times, stop[win_pos], side='left')

            if right <= left:
                continue

            matched = time_order[left:right]
            event_positions.append(matched)
            matched_windows.append(np.full(len(matched), win_pos, dtype=int))

        if not event_positions:
            return WindowMatches.build_empty()

        return WindowMatches(
            event_pos=np.concatenate(event_positions),
            win_pos=np.concatenate(matched_windows),
        )


def _assign_events_exclusive(
    windows: Windows,
    times: np.ndarray,
) -> np.ndarray:
    """Return one window position per event, or -1 when unmatched."""
    result = np.full(len(times), -1, dtype=int)
    win_positions = np.flatnonzero(~windows._is_empty())

    if len(times) == 0 or len(win_positions) == 0:
        return result

    start = windows._time_at('start')[win_positions]
    stop = windows._time_at('stop')[win_positions]
    order = np.lexsort((stop, start))
    start = start[order]
    stop = stop[order]
    win_positions = win_positions[order]

    slot = np.searchsorted(start, times, side='right') - 1
    candidate = slot >= 0
    event_pos = np.flatnonzero(candidate)
    slot = slot[candidate]
    inside = times[event_pos] < stop[slot]

    event_pos = event_pos[inside]
    slot = slot[inside]
    result[event_pos] = win_positions[slot]
    return result
