"""
Manage conversion between timescales & sampling rates, as well as defining windows of time
that can be used to cut data.
"""  # noqa: EXE002

from __future__ import annotations

import collections.abc
import dataclasses
import logging
import typing

import numpy as np
import pandas as pd

import nocte.core.collections
import nocte.core.time

logger = logging.getLogger(__name__)

WinPoint = typing.Literal['start', 'mid', 'ref', 'stop']

WinValue = float | np.ndarray | pd.Series


@dataclasses.dataclass(frozen=True, slots=True)
class Win:
    """
    A temporal window expressed in milliseconds.

    `start` and `stop` are offsets relative to `ref`, while `ref` locates
    the window in its enclosing temporal coordinate system.

    The represented interval is always half-open:

        [ref + start, ref + stop)

    All values are stored as float milliseconds. `ref` is always present
    and defaults to zero, which makes `Win(start, stop)` convenient for
    defining windows relative to an external event or marker.

    Examples
    --------
    A 30-minute window around an external reference:

        Win.in_units(-10, 20, 'minutes')

    The same geometry anchored at 2 hours:

        Win.in_units(-10, 20, 'minutes').shift(
            nocte.core.time.ms(hours=2)
        )

    A `(start, stop)` pair can be converted directly:

        pair = (-100.0, 500.0)
        win = Win(*pair)
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

    @classmethod
    def in_units(
        cls,
        start: float,
        stop: float,
        unit: nocte.core.time.TimeScale,
    ) -> typing.Self:
        """Build a window whose start and stop are expressed in the given unit."""
        scale = nocte.core.time.scale_to_ms(unit)

        return cls(
            start=float(start) * scale,
            stop=float(stop) * scale,
        )

    @classmethod
    def from_center(
        cls,
        center: float,
        duration: float,
        *,
        ref: float = 0.0,
    ) -> typing.Self:
        """Build a window of a given duration centered on a time point."""
        center = float(center)
        duration = float(duration)
        ref = float(ref)

        if duration < 0:
            raise ValueError('Duration must be non-negative')

        center_offset = center - ref
        half_duration = duration * 0.5

        return cls(
            center_offset - half_duration,
            center_offset + half_duration,
            ref=ref,
        )

    @property
    def length(self) -> float:
        """Window duration in milliseconds."""
        return self.stop - self.start

    @property
    def mid(self) -> float:
        """Midpoint of the window in the enclosing temporal coordinate."""
        return self.time_at(0.5)

    def is_empty(self) -> bool:
        """Return whether the window has zero duration."""
        return self.start == self.stop

    def time_at(self, q: float | WinPoint) -> float:
        """
        Return a time point in the enclosing temporal coordinate.

        A numeric value is interpreted as a fractional position through the
        window, where 0 is the start and 1 is the stop.
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
                typing.assert_never(q)

        q = float(q)

        if not 0.0 <= q <= 1.0:
            raise ValueError('Window position must be between 0 and 1')

        return self.ref + self.start + self.length * q

    def contains(
        self,
        t: float | np.ndarray,
    ) -> bool | np.ndarray:
        """Return whether time points fall within this `[start, stop)` window."""
        values = np.asarray(t)

        contained = (self.ref + self.start <= values) & (values < self.ref + self.stop)

        if contained.ndim == 0:
            return bool(contained.item())

        return contained

    def contained_in(self, other: typing.Self) -> bool:
        """Return whether this window is fully contained in another window."""
        return (
            other.ref + other.start <= self.ref + self.start
            and self.ref + self.stop <= other.ref + other.stop
        )

    def overlaps(self, other: typing.Self) -> bool:
        """Return whether this window and another window overlap."""
        start = max(
            self.ref + self.start,
            other.ref + other.start,
        )
        stop = min(
            self.ref + self.stop,
            other.ref + other.stop,
        )

        return start < stop

    def before(
        self,
        duration: float,
        *,
        offset: float = 0.0,
    ) -> typing.Self:
        """
        Return a window immediately before this one.

        Positive `offset` leaves a gap; negative `offset` creates overlap.
        """
        duration = float(duration)
        offset = float(offset)

        if duration < 0:
            raise ValueError('Duration must be non-negative')

        stop = self.start - offset

        return self.__class__(
            stop - duration,
            stop,
            ref=self.ref,
        )

    def after(
        self,
        duration: float,
        *,
        offset: float = 0.0,
    ) -> typing.Self:
        """
        Return a window immediately after this one.

        Positive `offset` leaves a gap; negative `offset` creates overlap.
        """
        duration = float(duration)
        offset = float(offset)

        if duration < 0:
            raise ValueError('Duration must be non-negative')

        start = self.stop + offset

        return self.__class__(
            start,
            start + duration,
            ref=self.ref,
        )

    def centered(
        self,
        duration: float,
        q: float | WinPoint = 'mid',
    ) -> typing.Self:
        """Return a window of the given duration centered on a point in this window."""
        return self.__class__.from_center(
            self.time_at(q),
            duration,
            ref=self.ref,
        )

    def change(
        self,
        pre: float = 0.0,
        post: float = 0.0,
    ) -> typing.Self:
        """
        Adjust the relative start and stop offsets.

        `pre` is added to `start` and `post` is added to `stop`.
        The reference is unchanged.
        """
        return dataclasses.replace(
            self,
            start=self.start + float(pre),
            stop=self.stop + float(post),
        )

    def shrink(self, duration: float = 0.0) -> typing.Self:
        """Shrink both sides by the same duration."""
        return self.change(+duration, -duration)

    def expand(self, duration: float = 0.0) -> typing.Self:
        """Expand both sides by the same duration."""
        return self.change(-duration, +duration)

    def shift(self, by: float = 0.0) -> typing.Self:
        """
        Shift the complete window in time without changing its geometry.

        Only the reference changes; start and stop remain unchanged.
        """
        return dataclasses.replace(
            self,
            ref=self.ref + float(by),
        )

    def crop(self, other: typing.Self) -> typing.Self:
        """
        Crop this window to another window.

        The returned window keeps this window's reference. If the windows do
        not overlap, the result is an empty window at the nearest boundary
        of `other`.
        """
        self_start = self.ref + self.start
        self_stop = self.ref + self.stop

        other_start = other.ref + other.start
        other_stop = other.ref + other.stop

        start = min(max(self_start, other_start), other_stop)
        stop = max(min(self_stop, other_stop), other_start)

        return self.__class__(
            start - self.ref,
            stop - self.ref,
            ref=self.ref,
        )

    def shift_to_fit(self, other: typing.Self) -> typing.Self:
        """
        Shift this window so it fits entirely within another window.

        The window duration and relative geometry are preserved.
        """
        if self.length > other.length:
            raise ValueError(f'Window {self} cannot fit within {other}')

        self_start = self.ref + self.start
        self_stop = self.ref + self.stop

        other_start = other.ref + other.start
        other_stop = other.ref + other.stop

        if self_start < other_start:
            return self.shift(other_start - self_start)

        if self_stop > other_stop:
            return self.shift(other_stop - self_stop)

        return self

    def take_centered(self, max_duration: float) -> typing.Self:
        """Return the centered portion of this window up to a maximum duration."""
        max_duration = float(max_duration)

        if max_duration < 0:
            raise ValueError('Maximum duration must be non-negative')

        return self.centered(min(self.length, max_duration))

    def arange(self, step: float) -> np.ndarray:
        """Return regularly spaced times across the half-open window."""
        step = float(step)

        if step <= 0:
            raise ValueError('Step must be positive')

        return np.arange(
            self.time_at('start'),
            self.time_at('stop'),
            step,
        )

    def round(
        self,
        decimals: int = 0,
        *,
        start: bool = True,
        stop: bool = True,
        scale: nocte.core.time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round relative start and/or stop offsets."""
        return dataclasses.replace(
            self,
            start=(
                nocte.core.time.ms_round(
                    self.start,
                    scale=scale,
                    decimals=decimals,
                )
                if start
                else self.start
            ),
            stop=(
                nocte.core.time.ms_round(
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
        scale: nocte.core.time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round relative start and/or stop offsets down."""
        return dataclasses.replace(
            self,
            start=(
                nocte.core.time.ms_floor(self.start, scale=scale)
                if start
                else self.start
            ),
            stop=(
                nocte.core.time.ms_floor(self.stop, scale=scale) if stop else self.stop
            ),
        )

    def ceil(
        self,
        *,
        start: bool = True,
        stop: bool = True,
        scale: nocte.core.time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round relative start and/or stop offsets up."""
        return dataclasses.replace(
            self,
            start=(
                nocte.core.time.ms_ceil(self.start, scale=scale)
                if start
                else self.start
            ),
            stop=(
                nocte.core.time.ms_ceil(self.stop, scale=scale) if stop else self.stop
            ),
        )

    def round_loose(
        self,
        scale: nocte.core.time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round start down and stop up."""
        return self.floor(
            start=True,
            stop=False,
            scale=scale,
        ).ceil(
            start=False,
            stop=True,
            scale=scale,
        )

    def round_tight(
        self,
        scale: nocte.core.time.TimeScale = 'milliseconds',
    ) -> typing.Self:
        """Round start up and stop down."""
        return self.ceil(
            start=True,
            stop=False,
            scale=scale,
        ).floor(
            start=False,
            stop=True,
            scale=scale,
        )


WinLike = Win | tuple[float, float]

TimeArrayLike = float | collections.abc.Sequence[float] | np.ndarray

MergeTake = typing.Literal['first', 'last']
SplitAlign = typing.Literal['left', 'right'] | float


class _WindowsData:
    """Internal immutable storage for window geometry."""

    _START = 0
    _STOP = 1
    _REF = 2

    def __init__(self, geometry: np.ndarray):
        geometry = np.ascontiguousarray(
            geometry,
            dtype=float,
        )

        if geometry.ndim != 2 or geometry.shape[1] != 3:
            raise ValueError('Window geometry must be a 2D array with 3 columns')

        if not np.isfinite(geometry).all():
            raise ValueError('Window geometry must be finite')

        if np.any(geometry[:, self._STOP] < geometry[:, self._START]):
            raise ValueError('Window stop must be greater than or equal to start')

        geometry.flags.writeable = False
        self._geometry = geometry

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

        start, stop, ref = [
            (np.full(n, value.item(), dtype=float) if value.ndim == 0 else value)
            for value in values
        ]

        return cls(
            np.column_stack(
                [
                    start,
                    stop,
                    ref,
                ]
            )
        )

    @property
    def start(self) -> np.ndarray:
        return self._geometry[:, self._START]

    @property
    def stop(self) -> np.ndarray:
        return self._geometry[:, self._STOP]

    @property
    def ref(self) -> np.ndarray:
        return self._geometry[:, self._REF]

    def __len__(self) -> int:
        return len(self._geometry)

    def take_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        return self.__class__(self._geometry[positions])

    def copy(self) -> typing.Self:
        return self.__class__(self._geometry.copy())


class Windows(nocte.core.collections.Collection):
    """
    Indexed collection of temporal windows.

    Temporal geometry is stored separately from item metadata as a float
    array with shape `(n_windows, 3)` and columns:

        start, stop, ref

    `start` and `stop` are offsets in milliseconds relative to `ref`.
    `ref` locates each window in the enclosing temporal coordinate system.

    Every represented interval is half-open:

        [ref + start, ref + stop)

    All geometry is finite float milliseconds. Sample-index windows are a
    separate concept and are not represented by this class.


    Empty windows are valid and retained everywhere.
    Creating a Windows containing empties emits a warning.
    Geometric operations apply the mathematically correct semantics of [t, t),
    rather than silently filtering those items out.
    """

    _RESERVED_META_COLUMNS = frozenset(
        {
            'start',
            'stop',
            'ref',
        }
    )

    def __init__(
        self,
        data: _WindowsData,
        meta: pd.DataFrame,
    ):
        self.data = data
        self.meta = meta.copy()

        reserved = self._RESERVED_META_COLUMNS.intersection(self.meta.columns)

        if reserved:
            raise ValueError(
                f'window geometry columns cannot appear in meta: {sorted(reserved)}'
            )

        self._validate_meta(len(self.data))

        n_empty = np.count_nonzero(self.is_empty())

        if n_empty:
            logger.warning(
                '%d/%d windows are empty',
                n_empty,
                len(self),
            )

    def _take_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        return self.__class__(
            self.data.take_pos(positions),
            self.meta.iloc[positions],
        )

    def copy(self) -> typing.Self:
        return self.__class__(
            self.data.copy(),
            self.meta.copy(),
        )

    def get(
        self,
        win_id: nocte.core.collections.ItemId | None = None,
    ) -> Win:
        """Return one window by public item identity."""
        if win_id is None:
            if len(self) != 1:
                raise ValueError(
                    'win_id is required unless there is exactly one window'
                )

            position = 0

        else:
            position = nocte.core.collections.ItemId(self._positions(win_id)[0])

        start = self.start[position]
        stop = self.stop[position]
        ref = self.ref[position]

        return Win(
            start,
            stop,
            ref=ref,
        )

    def items(self) -> collections.abc.Iterator[tuple[typing.Any, Win]]:
        """Iterate over `(window_id, Win)` pairs."""
        for win_id, start, stop, ref in zip(
            self.index, self.start, self.stop, self.ref, strict=True
        ):
            yield (
                win_id,
                Win(
                    start,
                    stop,
                    ref=ref,
                ),
            )

    @property
    def start(self) -> np.ndarray:
        """Start offsets relative to each window reference, in milliseconds."""
        return self.data.start

    @property
    def stop(self) -> np.ndarray:
        """Stop offsets relative to each window reference, in milliseconds."""
        return self.data.stop

    @property
    def ref(self) -> np.ndarray:
        """Window references in the enclosing temporal coordinate."""
        return self.data.ref

    @property
    def length(self) -> np.ndarray:
        """Window durations in milliseconds."""
        return self.stop - self.start

    @property
    def mid(self) -> np.ndarray:
        """Window midpoints in the enclosing temporal coordinate."""
        return self.time_at('mid')

    def time_at(self, q: float | WinPoint) -> np.ndarray:
        """
        Return one time point per window in the enclosing temporal coordinate.

        Numeric values are interpreted as fractional positions through each
        window, where 0 is the start and 1 is the stop.
        """
        if isinstance(q, str):
            if q == 'start':
                return self.ref + self.start

            if q == 'ref':
                return self.ref.copy()

            if q == 'stop':
                return self.ref + self.stop

            if q == 'mid':
                q = 0.5

            else:
                raise ValueError(f'Unknown window point: {q!r}')

        q = float(q)

        if not 0.0 <= q <= 1.0:
            raise ValueError('Window position must be between 0 and 1')

        return self.ref + self.start + self.length * q

    def contains(self, t: float) -> np.ndarray:
        """
        Return a mask indicating which windows contain a single time point.

        For classifying many time points, use the dedicated event matching
        machinery rather than constructing a window-by-time mask.
        """
        t = float(t)
        return (self.time_at('start') <= t) & (t < self.time_at('stop'))

    def contained_in(self, win: Win) -> np.ndarray:
        """Return a mask indicating which windows are fully contained in `win`."""
        win_start = win.ref + win.start
        win_stop = win.ref + win.stop

        return (win_start <= self.time_at('start')) & (self.time_at('stop') <= win_stop)

    def overlaps(self, win: Win) -> np.ndarray:
        """Return a mask indicating which windows overlap `win`."""
        start = np.maximum(
            self.time_at('start'),
            win.time_at('start'),
        )

        stop = np.minimum(
            self.time_at('stop'),
            win.time_at('stop'),
        )

        return start < stop

    def is_empty(self) -> np.ndarray:
        """Return a mask indicating zero-duration windows."""
        return self.start == self.stop

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
        Build windows from broadcastable start, stop, and ref values.

        Scalars are broadcast when possible.
        """

        data = _WindowsData.from_arrays(
            start=start,
            stop=stop,
            ref=ref,
        )

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
        """
        Instantiate the same relative window around multiple reference times.

        A pandas Series preserves its index, making this a one-to-one
        transformation from source identities to window identities.
        """
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
            mark_values = np.asarray(
                marks,
                dtype=float,
            )

            if mark_values.ndim == 0:
                mark_values = mark_values.reshape(1)

            elif mark_values.ndim != 1:
                raise ValueError('marks must be scalar or one-dimensional')

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
            Win.from_center(
                0.0,
                duration,
            ),
            meta=meta,
        )

    @classmethod
    def build_between(
        cls,
        times: TimeArrayLike | pd.Series,
    ) -> typing.Self:
        """
        Build windows between successive time points.

        Each result uses its left edge as its reference:

            start = 0
            ref = left edge
            stop = right edge - left edge

        If `times` is a Series, the source identities of both bounding
        markers are retained explicitly in metadata.
        """
        source_ids = None

        if isinstance(times, pd.Series):
            values = times.to_numpy(dtype=float)
            source_ids = times.index.to_numpy()
            source_name = times.index.name or 'source_id'

        else:
            values = np.asarray(
                times,
                dtype=float,
            )
            source_name = None

        if values.ndim == 0:
            values = values.reshape(1)

        elif values.ndim != 1:
            raise ValueError('times must be scalar or one-dimensional')

        order = np.argsort(
            values,
            kind='stable',
        )

        values = values[order]

        if source_ids is not None:
            source_ids = source_ids[order]

        if len(values) < 2:
            return cls.from_arrays(
                [],
                [],
            )

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
        wins: collections.abc.Mapping[
            typing.Any,
            WinLike,
        ],
        *,
        name: str = 'cat',
    ) -> typing.Self:
        """Build windows from a mapping of metadata values to definitions."""
        labels = []
        start = []
        stop = []
        ref = []

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
            meta=pd.DataFrame(
                {
                    name: labels,
                }
            ),
        )

    def around(
        self,
        win: WinLike,
        q: float | WinPoint = 'mid',
    ) -> typing.Self:
        """Build a new window around one point from each current window."""
        marks = pd.Series(
            self.time_at(q),
            index=self.index,
        )

        return self.__class__.build_around(
            marks,
            win,
            meta=self.meta,
        )

    def centered(
        self,
        duration: float,
        q: float | WinPoint = 'mid',
    ) -> typing.Self:
        """Build equal-duration windows centered on a point in each window."""
        return self.around(
            Win.from_center(
                0.0,
                duration,
            ),
            q=q,
        )

    def before(
        self,
        duration: float,
        *,
        offset: float = 0.0,
        q: float | WinPoint = 'start',
    ) -> typing.Self:
        """
        Build windows before a point in each current window.

        Positive offset leaves a gap.
        """
        duration = float(duration)
        offset = float(offset)

        if duration < 0:
            raise ValueError('Duration must be non-negative')

        return self.around(
            Win(
                -duration - offset,
                -offset,
            ),
            q=q,
        )

    def after(
        self,
        duration: float,
        *,
        offset: float = 0.0,
        q: float | WinPoint = 'stop',
    ) -> typing.Self:
        """
        Build windows after a point in each current window.

        Positive offset leaves a gap.
        """
        duration = float(duration)
        offset = float(offset)

        if duration < 0:
            raise ValueError('Duration must be non-negative')

        return self.around(
            Win(
                offset,
                duration + offset,
            ),
            q=q,
        )

    def change(self, pre: WinValue = 0.0, post: WinValue = 0.0) -> typing.Self:
        pre = self._broadcast_value(pre, name='pre')
        post = self._broadcast_value(post, name='post')

        return self.__class__.from_arrays(
            self.start + pre,
            self.stop + post,
            self.ref,
            meta=self.meta,
        )

    def shrink(self, duration: WinValue = 0.0) -> typing.Self:
        """Shrink both sides by the same duration."""
        return self.change(
            +duration,
            -duration,
        )

    def expand(self, duration: WinValue = 0.0) -> typing.Self:
        """Expand both sides by the same duration."""
        return self.change(
            -duration,
            +duration,
        )

    def _broadcast_value(
        self,
        values: float | np.ndarray | pd.Series,
        *,
        name: str,
    ) -> np.ndarray:
        """Get a one-dimensional array of values, broadcasting scalars to match the number of windows."""
        if isinstance(values, pd.Series):
            values = values.reindex(self.index)

            if values.isna().any():
                raise ValueError(f'{name} is missing values for some windows')

            result = values.to_numpy(dtype=float)

        else:
            result = np.asarray(
                values,
                dtype=float,
            )

            if result.ndim == 0:
                result = np.full(
                    len(self),
                    float(result),
                )

            elif result.ndim != 1 or len(result) != len(self):
                raise ValueError(f'{name} must be scalar or have one value per window')

        if not np.isfinite(result).all():
            raise ValueError(f'{name} must be finite')

        return result

    def shift(self, by: WinValue = 0.0) -> typing.Self:
        """
        Shift windows in time.

        Relative geometry is preserved; only references move.
        """
        by = self._broadcast_value(by, name='shift')

        return self.__class__.from_arrays(
            self.start,
            self.stop,
            self.ref + by,
            meta=self.meta,
        )

    def reanchor(
        self,
        q: float | WinPoint,
    ) -> typing.Self:
        """
        Change each reference while preserving the represented intervals.
        """
        new_ref = self.time_at(q)
        delta = new_ref - self.ref

        return self.__class__.from_arrays(
            self.start - delta,
            self.stop - delta,
            new_ref,
            meta=self.meta,
        )

    def crop(self, win: Win) -> typing.Self:
        """
        Crop every window to `win`.

        Window identities and references are preserved. Windows completely
        outside `win` become empty rather than being dropped.
        """
        start = self.time_at('start')
        stop = self.time_at('stop')

        win_start = win.ref + win.start
        win_stop = win.ref + win.stop

        cropped_start = np.minimum(
            np.maximum(
                start,
                win_start,
            ),
            win_stop,
        )

        cropped_stop = np.maximum(
            np.minimum(
                stop,
                win_stop,
            ),
            win_start,
        )

        return self.__class__.from_arrays(
            cropped_start - self.ref,
            cropped_stop - self.ref,
            self.ref,
            meta=self.meta,
        )

    def drop_empty(
        self,
    ) -> typing.Self:
        """Drop zero-duration windows."""
        return self.sel_mask(~self.is_empty())

    # ------------------------------------------------------------------
    # geometry properties

    def are_uniform(self, atol: float = 1e-8) -> bool:
        """Return whether all windows share the same relative geometry."""
        if len(self) <= 1:
            return True

        return bool(
            np.allclose(
                self.start,
                self.start[0],
                atol=atol,
            )
            and np.allclose(
                self.stop,
                self.stop[0],
                atol=atol,
            )
        )

    def _sorted_realized_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return realized start and stop times, sorted by time.

        `start` and `stop` are stored relative to each window's reference, so this
        helper first converts them to the enclosing temporal coordinate:

            realized_start = ref + start
            realized_stop = ref + stop

        Windows are sorted by realized start time, using stop time as a
        secondary key. Empty windows are retained.
        """
        start = self.time_at('start')
        stop = self.time_at('stop')

        order = np.lexsort(
            (
                stop,
                start,
            )
        )

        return start[order], stop[order]

    def are_exclusive(self) -> bool:
        """Return whether non-empty windows do not overlap."""
        start, stop = self._sorted_realized_bounds()

        if len(start) <= 1:
            return True

        previous_stop = np.maximum.accumulate(stop[:-1])

        return bool(np.all(start[1:] >= previous_stop))

    def are_tight(self) -> bool:
        """
        Return whether non-empty windows leave no gaps in their coverage.

        Overlapping windows can therefore be tight without being exclusive.
        """
        start, stop = self._sorted_realized_bounds()

        if len(start) <= 1:
            return True

        previous_stop = np.maximum.accumulate(stop[:-1])

        return bool(np.all(start[1:] <= previous_stop))

    def total(self) -> float:
        """Return the sum of all window durations."""
        return float(self.length.sum())

    def global_win(self) -> Win:
        """Return the minimum window containing all windows."""
        if len(self) == 0:
            raise ValueError('cannot get a global window from an empty collection')

        start = float(self.time_at('start').min())

        stop = float(self.time_at('stop').max())

        return Win(
            0.0,
            stop - start,
            ref=start,
        )

    def sort_time(self) -> typing.Self:
        """Sort windows by realized start, then stop time."""
        positions = np.lexsort(
            (
                self.time_at('stop'),
                self.time_at('start'),
            )
        )

        return self._take_pos(positions)

    def to_frame(self) -> pd.DataFrame:
        """
        Return temporal geometry and metadata together.

        `start` and `stop` remain relative offsets; this is a view for
        inspection/export, not the internal representation.
        """
        df = self.meta.copy()

        df.insert(0, 'ref', self.ref)
        df.insert(0, 'stop', self.stop)
        df.insert(0, 'start', self.start)

        return df

    def _repr_html_(self) -> str:
        df = self.to_frame()

        return df._repr_html_()  # type: ignore

    def edges(self) -> np.ndarray:
        """Return sorted unique realized window edges."""
        start, stop = self._sorted_realized_bounds()

        return np.unique(np.concatenate([start, stop]))

    def breaks(self) -> np.ndarray:
        """
        Return the ordered boundaries of a tight, exclusive set of windows.

        For windows `[a, b), [b, c), [c, d)`, returns `[a, b, c, d]`.
        Empty windows are ignored.
        """
        if not self.are_exclusive() or not self.are_tight():
            raise ValueError('breaks require windows to be both tight and exclusive')

        start, stop = self._sorted_realized_bounds()

        if len(start) == 0:
            return np.empty(0, dtype=float)

        return np.concatenate([start, stop[-1:]])

    def _merge_connected(
        self,
        *,
        touching: bool,
        take: MergeTake,
    ) -> typing.Self:
        """
        Merge overlapping or connected windows.

        If `touching` is False, only windows with a positive-duration
        intersection are merged. If True, touching windows are merged as well.

        Empty windows are retained as standalone items.

        Metadata and reference are taken from either the first or last source
        window in each merged group. Contributing item IDs are recorded in
        `source_win_ids`.
        """

        if take not in ('first', 'last'):
            raise ValueError(f'Unknown take policy: {take!r}')

        start = self.time_at('start')
        stop = self.time_at('stop')

        nonempty = np.flatnonzero(start < stop)
        empty = np.flatnonzero(start == stop)

        order = nonempty[np.argsort(start[nonempty], kind='stable')]

        groups: list[list[int]] = []

        for pos in order:
            pos = int(pos)

            if not groups:
                groups.append([pos])
                continue

            group = groups[-1]
            group_stop = max(stop[group])

            connected = (
                start[pos] <= group_stop if touching else start[pos] < group_stop
            )

            if connected:
                group.append(pos)
            else:
                groups.append([pos])

        groups.extend([[int(pos)] for pos in empty])
        groups.sort(key=lambda group: start[group[0]])

        rows = []
        starts = []
        stops = []
        refs = []

        for group in groups:
            source = group[0] if take == 'first' else group[-1]
            ref = self.ref[source]

            starts.append(min(start[group]) - ref)
            stops.append(max(stop[group]) - ref)
            refs.append(ref)

            row = self.meta.iloc[source].copy()
            row['source_win_ids'] = tuple(self.index[group])
            rows.append(row)

        meta = pd.DataFrame(rows)
        meta.index = pd.RangeIndex(len(meta), name=self.index.name)

        return self.from_arrays(
            starts,
            stops,
            refs,
            meta=meta,
        )

    def merge_overlap(
        self,
        *,
        take: MergeTake = 'first',
    ) -> typing.Self:
        """Merge overlapping windows."""
        return self._merge_connected(
            touching=False,
            take=take,
        )

    def merge_tight(
        self,
        *,
        take: MergeTake = 'first',
    ) -> typing.Self:
        """Merge overlapping or touching windows into continuous regions."""
        return self._merge_connected(
            touching=True,
            take=take,
        )

    def invert(
        self,
        within: Win,
    ) -> typing.Self:
        """
        Return all uncovered regions inside `within`.

        Input windows are treated as coverage, so overlaps and touching regions
        are implicitly combined. Result windows use their left edge as reference.
        """
        outer_start = within.ref + within.start
        outer_stop = within.ref + within.stop

        if outer_start == outer_stop:
            return self.__class__.from_arrays(
                [],
                [],
                [],
            )

        start = self.time_at('start')
        stop = self.time_at('stop')

        valid = ~self.is_empty() & (start < outer_stop) & (outer_start < stop)

        start = np.maximum(
            start[valid],
            outer_start,
        )

        stop = np.minimum(
            stop[valid],
            outer_stop,
        )

        if len(start) == 0:
            return self.__class__.from_arrays(
                0.0,
                outer_stop - outer_start,
                outer_start,
            )

        order = np.lexsort((stop, start))

        start = start[order]
        stop = stop[order]

        gap_start = []
        gap_stop = []

        cursor = outer_start

        for this_start, this_stop in zip(
            start,
            stop,
            strict=True,
        ):
            if this_start > cursor:
                gap_start.append(cursor)
                gap_stop.append(this_start)

            cursor = max(
                cursor,
                this_stop,
            )

            if cursor >= outer_stop:
                break

        if cursor < outer_stop:
            gap_start.append(cursor)
            gap_stop.append(outer_stop)

        gap_start = np.asarray(
            gap_start,
            dtype=float,
        )
        gap_stop = np.asarray(
            gap_stop,
            dtype=float,
        )

        return self.__class__.from_arrays(
            start=np.zeros(len(gap_start)),
            stop=gap_stop - gap_start,
            ref=gap_start,
        )

    def split(
        self,
        length: float,
        *,
        align: SplitAlign = 'left',
    ) -> typing.Self:
        """
        Split each window into equal-length fragments.

        Only complete fragments are returned. Any remainder is distributed
        according to `align`:

            'left'  / 0.0 -> remainder on the right
            'right' / 1.0 -> remainder on the left

        Numeric alignment values between 0 and 1 interpolate between these.

        Metadata is copied from each source window. `source_win_id` and
        `fragment_idx` record provenance.
        """
        length = float(length)

        if not np.isfinite(length) or length <= 0:
            raise ValueError('Split length must be finite and positive')

        if isinstance(align, str):
            try:
                align_value = {
                    'left': 0.0,
                    'right': 1.0,
                }[align]
            except KeyError:
                raise ValueError(f'Unknown alignment: {align!r}') from None

        else:
            align_value = float(align)

            if not 0.0 <= align_value <= 1.0:
                raise ValueError('Numeric alignment must be between 0 and 1')

        for col in (
            'source_win_id',
            'fragment_idx',
        ):
            if col in self.meta.columns:
                raise ValueError(f'meta already contains {col!r}')

        starts = []
        stops = []
        refs = []
        source_positions = []
        fragment_idcs = []

        for pos in range(len(self)):
            ratio = self.length[pos] / length

            if np.isclose(
                ratio,
                round(ratio),
            ):
                n = round(ratio)
            else:
                n = int(np.floor(ratio))

            if n == 0:
                continue

            remainder = self.length[pos] - n * length

            if np.isclose(remainder, 0):
                remainder = 0.0

            offset = remainder * align_value

            fragment_start = self.start[pos] + offset + np.arange(n) * length

            starts.append(fragment_start)
            stops.append(fragment_start + length)
            refs.append(
                np.full(
                    n,
                    self.ref[pos],
                )
            )

            source_positions.extend([pos] * n)
            fragment_idcs.extend(range(n))

        source_positions = np.asarray(
            source_positions,
            dtype=int,
        )

        meta = self.meta.iloc[source_positions].copy()

        meta.insert(
            0,
            'source_win_id',
            self.index.to_numpy()[source_positions],
        )

        meta.insert(
            1,
            'fragment_idx',
            fragment_idcs,
        )

        meta.index = pd.RangeIndex(
            len(meta),
            name='win_id',
        )

        if not starts:
            return self.__class__.from_arrays(
                [],
                [],
                [],
                meta=meta,
            )

        return self.__class__.from_arrays(
            start=np.concatenate(starts),
            stop=np.concatenate(stops),
            ref=np.concatenate(refs),
            meta=meta,
        )

    def defrag(
        self,
        start: float = 0.0,
    ) -> typing.Self:
        """
        Shift windows so they form a tight sequence in their current item order.

        Durations, relative geometry, metadata, and item identities are preserved.
        Only references are shifted.
        """
        start = float(start)

        if not np.isfinite(start):
            raise ValueError('Defragmentation start must be finite')

        if len(self) == 0:
            return self.copy()

        target_start = start + np.concatenate(
            [
                [0.0],
                np.cumsum(self.length[:-1]),
            ]
        )

        shift = target_start - self.time_at('start')

        return self.shift(shift)

    def _time_order(self) -> np.ndarray:
        """Return positions sorted by realized start, then stop."""
        return np.lexsort(
            (
                self.time_at('stop'),
                self.time_at('start'),
            )
        )

    def interval_to_prev(
        self,
        shift: int = 1,
    ) -> np.ndarray:
        """
        Return the interval from each window to the previous window in time.

        Positive values indicate a gap; zero means touching; negative values
        indicate overlap. Windows without a previous neighbor return infinity.
        """
        if shift < 1:
            raise ValueError('shift must be at least 1')

        result = np.full(
            len(self),
            np.inf,
        )

        if shift >= len(self):
            return result

        start = self.time_at('start')
        stop = self.time_at('stop')
        order = self._time_order()

        current = order[shift:]
        previous = order[:-shift]

        result[current] = start[current] - stop[previous]

        return result

    def interval_to_next(
        self,
        shift: int = 1,
    ) -> np.ndarray:
        """
        Return the interval from each window to the next window in time.

        Positive values indicate a gap; zero means touching; negative values
        indicate overlap. Windows without a next neighbor return infinity.
        """
        if shift < 1:
            raise ValueError('shift must be at least 1')

        result = np.full(
            len(self),
            np.inf,
        )

        if shift >= len(self):
            return result

        start = self.time_at('start')
        stop = self.time_at('stop')
        order = self._time_order()

        current = order[:-shift]
        following = order[shift:]

        result[current] = start[following] - stop[current]

        return result

    def interval_to_closest(self) -> np.ndarray:
        """Return the smaller interval to the previous or next window."""
        return np.minimum(
            self.interval_to_prev(),
            self.interval_to_next(),
        )

    def is_isolated(
        self,
        at_least: float | tuple[float, float],
    ) -> np.ndarray:
        """
        Return whether windows have at least the requested separation.

        A scalar applies equally before and after. A `(pre, post)` tuple allows
        asymmetric requirements.
        """
        if isinstance(at_least, tuple):
            pre, post = at_least
        else:
            pre = post = at_least

        pre = float(pre)
        post = float(post)

        if pre < 0 or post < 0:
            raise ValueError('Isolation thresholds must be non-negative')

        return (self.interval_to_prev() >= pre) & (self.interval_to_next() >= post)

    def classify_events(
        self,
        times: TimeArrayLike | pd.Series,
        *,
        relative_to: float | WinPoint = 'ref',
        merge_meta: str | collections.abc.Iterable[str] = (),
    ) -> pd.DataFrame:
        """
        Classify events into windows.

        Windows always follow `[start, stop)` semantics. Events outside all
        windows are omitted. If windows overlap, an event may occur in multiple
        output rows.

        `delay` is measured relative to `relative_to` within the matched window.
        """
        if isinstance(times, pd.Series):
            values = times.to_numpy(dtype=float)
            event_index = times.index

        else:
            values = np.asarray(
                times,
                dtype=float,
            )

            if values.ndim == 0:
                values = values.reshape(1)

            elif values.ndim != 1:
                raise ValueError('times must be scalar or one-dimensional')

            event_index = pd.RangeIndex(
                len(values),
                name='event_idx',
            )

        if isinstance(merge_meta, str):
            merge_meta = [merge_meta]
        else:
            merge_meta = list(merge_meta)

        missing = set(merge_meta).difference(self.meta.columns)

        if missing:
            raise KeyError(f'Unknown metadata columns: {sorted(missing)}')

        if self.are_exclusive():
            event_pos, win_pos = _classify_events_exclusive(
                self,
                values,
            )

        else:
            event_pos, win_pos = _classify_events_overlapping(
                self,
                values,
            )

        anchor = self.time_at(relative_to)

        win_col = self.index.name or 'win_id'

        result = pd.DataFrame(
            {
                win_col: self.index.to_numpy()[win_pos],
                'delay': values[event_pos] - anchor[win_pos],
            },
            index=event_index.take(event_pos),
        )

        for col in merge_meta:
            result[col] = self.meta[col].to_numpy()[win_pos]

        return result


def _classify_events_exclusive(
    windows: Windows,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match events to exclusive windows.

    Returns parallel arrays `(event_positions, window_positions)`.
    """
    win_positions = np.flatnonzero(~windows.is_empty())

    if len(times) == 0 or len(win_positions) == 0:
        empty = np.empty(
            0,
            dtype=int,
        )
        return empty, empty

    start = windows.time_at('start')[win_positions]

    stop = windows.time_at('stop')[win_positions]

    order = np.lexsort((stop, start))

    start = start[order]
    stop = stop[order]
    win_positions = win_positions[order]

    slot = (
        np.searchsorted(
            start,
            times,
            side='right',
        )
        - 1
    )

    candidate = slot >= 0

    event_positions = np.flatnonzero(candidate)

    slot = slot[candidate]

    inside = times[event_positions] < stop[slot]

    event_positions = event_positions[inside]

    win_positions = win_positions[slot[inside]]

    return (
        event_positions,
        win_positions,
    )


def _classify_events_overlapping(
    windows: Windows,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match events to potentially overlapping windows.

    An event may appear more than once in the result.
    """
    if len(times) == 0:
        empty = np.empty(
            0,
            dtype=int,
        )
        return empty, empty

    time_order = np.argsort(
        times,
        kind='stable',
    )

    sorted_times = times[time_order]

    start = windows.time_at('start')
    stop = windows.time_at('stop')

    event_positions = []
    win_positions = []

    for win_pos in np.flatnonzero(~windows.is_empty()):
        left = np.searchsorted(
            sorted_times,
            start[win_pos],
            side='left',
        )

        right = np.searchsorted(
            sorted_times,
            stop[win_pos],
            side='left',
        )

        if right <= left:
            continue

        matched = time_order[left:right]

        event_positions.append(matched)

        win_positions.append(
            np.full(
                len(matched),
                win_pos,
                dtype=int,
            )
        )

    if not event_positions:
        empty = np.empty(
            0,
            dtype=int,
        )
        return empty, empty

    event_positions = np.concatenate(event_positions)

    win_positions = np.concatenate(win_positions)

    order = np.lexsort(
        (
            win_positions,
            event_positions,
        )
    )

    return (
        event_positions[order],
        win_positions[order],
    )
