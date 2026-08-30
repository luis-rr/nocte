"""Collections of related point-event trains."""

from __future__ import annotations

import collections.abc
import itertools
import pathlib
import typing

import h5py
import numpy as np
import pandas as pd

import nocte.core._point_process
import nocte.core.hdf
import nocte.core.time
import nocte.core.windows

TrainArrayLike = collections.abc.Sequence[float] | np.ndarray | pd.Index
TrainValuesLike = (
    collections.abc.Sequence[TrainArrayLike]
    | collections.abc.Mapping[int, TrainArrayLike]
)


class _TrainsData:
    """
    Internal immutable ragged storage for timestamp trains.

    One contiguous read-only NumPy array is stored per train. This keeps train
    selection cheap, avoids a long spike-level DataFrame, and lets numerical
    operations work directly on already-separated sorted timestamps.
    """

    def __init__(self, values: collections.abc.Sequence[TrainArrayLike]):
        arrays = []

        for times in values:
            array = nocte.core._point_process.as_sorted_times_1d(times, copy=True)
            array.flags.writeable = False
            arrays.append(array)

        self.values = tuple(arrays)

    @classmethod
    def _from_readonly_arrays(
        cls,
        values: collections.abc.Sequence[np.ndarray],
    ) -> typing.Self:
        """Construct from already validated read-only arrays without copying."""
        obj = cls.__new__(cls)
        obj.values = tuple(values)
        return obj

    def __len__(self) -> int:
        return len(self.values)

    def get_pos(self, position: int) -> np.ndarray:
        """Return one train as a read-only timestamp array."""
        return self.values[position]

    def sel_pos(self, positions: np.ndarray) -> typing.Self:
        arrays = [self.values[int(position)] for position in positions]
        return self.__class__._from_readonly_arrays(arrays)

    def crop(self, start: float, stop: float) -> typing.Self:
        """Restrict every train to one half-open interval."""
        arrays = []

        for times in self.values:
            left = np.searchsorted(times, start, side='left')
            right = np.searchsorted(times, stop, side='left')
            arrays.append(times[left:right])

        return self.__class__._from_readonly_arrays(arrays)

    def shift(self, by: float) -> typing.Self:
        arrays = []

        for times in self.values:
            shifted = np.ascontiguousarray(times + by, dtype=float)
            shifted.flags.writeable = False
            arrays.append(shifted)

        return self.__class__._from_readonly_arrays(arrays)

    def counts(self) -> np.ndarray:
        return np.fromiter(
            (len(times) for times in self.values),
            dtype=np.int64,
            count=len(self),
        )

    def flatten(self) -> tuple[np.ndarray, np.ndarray]:
        """Return flat timestamps and cumulative offsets for serialization."""
        counts = self.counts()
        offsets = np.empty(len(self) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        if offsets[-1] == 0:
            flat = np.empty(0, dtype=float)
        else:
            flat = np.concatenate(self.values)

        return flat, offsets

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        """Store trains as flat timestamps plus cumulative offsets."""
        key = nocte.core.hdf.normalize_hdf_key(key)
        flat, offsets = self.flatten()

        with h5py.File(path, mode='a') as file:
            if key in file:
                raise FileExistsError(f'HDF5 key {key!r} already exists')

            group = file.create_group(key)
            group.create_dataset('times', data=flat)
            group.create_dataset('offsets', data=offsets)

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> typing.Self:
        """Load trains stored as flat timestamps plus cumulative offsets."""
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='r') as file:
            if key not in file:
                raise KeyError(f'HDF5 key {key!r} does not exist')

            node = file[key]

            if not isinstance(node, h5py.Group):
                raise TypeError(f'Trains payload {key!r} must be an HDF5 group')

            times_node = node.get('times')
            offsets_node = node.get('offsets')

            if not isinstance(times_node, h5py.Dataset):
                raise TypeError('Trains payload times must be an HDF5 dataset')

            if not isinstance(offsets_node, h5py.Dataset):
                raise TypeError('Trains payload offsets must be an HDF5 dataset')

            flat = np.asarray(times_node[...], dtype=float)
            offsets = np.asarray(offsets_node[...], dtype=np.int64)

        if offsets.ndim != 1 or len(offsets) == 0:
            raise ValueError('Trains offsets must be a non-empty 1D array')

        if offsets[0] != 0 or offsets[-1] != len(flat):
            raise ValueError('Trains offsets do not span the flat timestamp array')

        if np.any(offsets[1:] < offsets[:-1]):
            raise ValueError('Trains offsets must be monotonically non-decreasing')

        # Copy each train once so selecting one loaded train does not retain the
        # complete flat HDF payload in memory.
        arrays = []

        for start, stop in itertools.pairwise(offsets):
            array = np.array(
                flat[start:stop],
                dtype=float,
                order='C',
                copy=True,
            )
            array = nocte.core._point_process.as_sorted_times_1d(array)
            array.flags.writeable = False
            arrays.append(array)

        return cls._from_readonly_arrays(arrays)


class Trains(nocte.core.hdf.HDFCollection[np.ndarray]):
    """
    Indexed collection of related point-event trains.

    The primary collection item is one train, commonly the spike train of one
    unit. Each item owns a sorted variable-length sequence of finite timestamps
    in milliseconds. Metadata has one row per train, not one row per timestamp.

    ``support`` is the collection-wide half-open temporal interval over which
    all trains were observed. Empty trains are valid and remain meaningful
    because their observation support is explicit.
    """

    def __init__(
        self,
        data: _TrainsData,
        meta: pd.DataFrame,
        support: nocte.core.windows.Win,
    ):
        self._data = data
        self.meta = meta.copy()
        self.support = support

        if self.meta.index.name is None:
            self.meta.index = self.meta.index.rename('train_id')

        self._validate_meta(len(self._data))
        self._validate_support()

    # ------------------------------------------------------------------
    # core collection and access

    @staticmethod
    def _default_meta(n_items: int) -> pd.DataFrame:
        return pd.DataFrame(
            index=pd.RangeIndex(n_items, name='train_id'),
        )

    def _sel_pos(self, positions: np.ndarray) -> typing.Self:
        return self.__class__(
            self._data.sel_pos(positions),
            self.meta.iloc[positions],
            self.support,
        )

    def _get_pos(self, position: int) -> np.ndarray:
        return self._data.get_pos(position)

    def copy(self) -> typing.Self:
        """Return a collection sharing immutable timestamps and copying metadata."""
        return self.__class__(self._data, self.meta, self.support)

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def from_times(
        cls,
        times: TrainValuesLike,
        *,
        support: nocte.core.windows.Win,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        """
        Build trains from timestamp arrays.

        A mapping uses its integer keys as train identities when ``meta`` is
        omitted. With explicit metadata, mapping keys must match ``meta.index``.
        A sequence is interpreted positionally.
        """
        if isinstance(times, collections.abc.Mapping):
            keys = pd.Index(times.keys())

            if not pd.api.types.is_integer_dtype(keys.dtype):
                raise ValueError('Train mapping keys must be integers')
            if not keys.is_unique:
                raise ValueError('Train mapping keys must be unique')

            if meta is None:
                meta = pd.DataFrame(index=keys.copy())
                arrays = list(times.values())
            else:
                missing = meta.index.difference(keys)
                extra = keys.difference(meta.index)
                if len(missing) or len(extra):
                    raise ValueError('Train mapping keys must match meta index exactly')
                arrays = [times[int(train_id)] for train_id in meta.index]
        else:
            arrays = list(times)
            if meta is None:
                meta = cls._default_meta(len(arrays))

        data = _TrainsData(arrays)
        return cls(data, meta, support)

    # ------------------------------------------------------------------
    # basic train summaries

    def counts(self) -> pd.Series:
        """Number of point events in each train."""
        return pd.Series(
            self._data.counts(),
            index=self.index,
            name='count',
        )

    def rates(self) -> pd.Series:
        """Mean rate of each train across collection support, in Hz."""
        if self.support.length == 0:
            raise ValueError('Mean rate is undefined for empty temporal support')

        rate_hz = self._data.counts() / self.support.length
        rate_hz *= nocte.core.time.ms(seconds=1)

        return pd.Series(
            rate_hz,
            index=self.index,
            name='rate',
        )

    def intervals(self) -> dict[int, np.ndarray]:
        """Return successive within-train intervals in milliseconds."""
        return {int(train_id): np.diff(times) for train_id, times in self.items()}

    def drop_silent(self) -> typing.Self:
        """Drop trains containing no point events."""
        return self.sel_mask(self.counts() > 0)

    # ------------------------------------------------------------------
    # temporal transformations

    def crop(self, win: nocte.core.windows.Win) -> typing.Self:
        """
        Restrict all trains to the intersection of ``support`` and ``win``.

        Train identities are preserved, including trains that become empty.
        """
        support = self.support.crop(win)
        start = support.time_at('start')
        stop = support.time_at('stop')

        return self.__class__(
            self._data.crop(start, stop),
            self.meta,
            support,
        )

    def shift(self, by: float) -> typing.Self:
        """Shift every timestamp and collection support by the same amount."""
        by = float(by)
        if not np.isfinite(by):
            raise ValueError('Shift must be finite')

        return self.__class__(
            self._data.shift(by),
            self.meta,
            self.support.shift(by),
        )

    # ------------------------------------------------------------------
    # point-process summaries over common support

    def count_in(self, win: nocte.core.windows.Win) -> pd.Series:
        """Count events from each train in one observed half-open interval."""
        self._require_within_support(win)
        start = win.time_at('start')
        stop = win.time_at('stop')

        values = nocte.core._point_process.count_between_many(
            self._data.values,
            start,
            stop,
        )

        return pd.Series(values, index=self.index, name='count')

    def rate_in(self, win: nocte.core.windows.Win) -> pd.Series:
        """Mean rate of each train inside one observed interval, in Hz."""
        if win.length == 0:
            raise ValueError('Rate is undefined for an empty window')

        counts = self.count_in(win).astype(float)
        counts *= nocte.core.time.ms(seconds=1) / win.length
        counts.name = 'rate'
        return counts

    def count_bins(
        self,
        bins: nocte.core._point_process.TimeArrayLike,
    ) -> pd.DataFrame:
        """
        Count every train in common half-open bins ``[left, right)``.

        Bins must lie inside known observation support so unobserved time is not
        silently represented as zero activity.
        """
        edges = nocte.core._point_process.as_bin_edges(bins)
        self._require_range_within_support(edges[0], edges[-1], name='bins')

        values = nocte.core._point_process.count_bins_many(
            self._data.values,
            edges,
        )

        return pd.DataFrame(
            values.T,
            index=pd.IntervalIndex.from_breaks(edges, closed='left', name='time'),
            columns=self.index.copy(),
        )

    def count_rolling(
        self,
        window: float,
        *,
        step: float,
        within: nocte.core.windows.Win | None = None,
    ) -> pd.DataFrame:
        """Count every train in centered half-open sliding windows."""
        window = self._positive_float(window, name='window')
        step = self._positive_float(step, name='step')
        within = self.support if within is None else within
        self._require_within_support(within)

        sample_times = nocte.core._point_process.sample_centers(
            within.time_at('start'),
            within.time_at('stop'),
            step,
            margin=window * 0.5,
        )
        values = nocte.core._point_process.count_rolling_many(
            self._data.values,
            sample_times,
            window,
        )

        return pd.DataFrame(
            values.T,
            index=pd.Index(sample_times, name='time'),
            columns=self.index.copy(),
        )

    def rate_gaussian(
        self,
        sigma: float,
        *,
        step: float,
        within: nocte.core.windows.Win | None = None,
        width: float = 5.0,
    ) -> pd.DataFrame:
        """
        Estimate Gaussian-smoothed rate for every train, in Hz.

        The shared point-process helper owns the per-train loop. Its expensive
        single-train kernel is Numba compiled and parallelizes over sample times;
        no long event table is constructed.
        """
        sigma = self._positive_float(sigma, name='sigma')
        step = self._positive_float(step, name='step')
        width = self._positive_float(width, name='width')
        within = self.support if within is None else within
        self._require_within_support(within)

        sample_times = nocte.core._point_process.sample_centers(
            within.time_at('start'),
            within.time_at('stop'),
            step,
            margin=sigma * width,
        )
        values = nocte.core._point_process.gaussian_rate_many(
            self._data.values,
            sample_times,
            sigma,
            width,
        )
        values *= nocte.core.time.ms(seconds=1)

        return pd.DataFrame(
            values.T,
            index=pd.Index(sample_times, name='time'),
            columns=self.index.copy(),
        )

    # ------------------------------------------------------------------
    # serialization

    def _to_hdf_data(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        data_key = f'{key}/data'
        self._data.to_hdf(path, key=data_key)

        with h5py.File(path, mode='a') as file:
            node = file[data_key]
            if not isinstance(node, h5py.Group):
                raise TypeError(f'Trains payload {data_key!r} must be a group')

            node.create_dataset(
                'support',
                data=np.asarray(
                    [self.support.start, self.support.stop, self.support.ref],
                    dtype=float,
                ),
            )

    @classmethod
    def _from_hdf_data(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        meta: pd.DataFrame,
    ) -> typing.Self:
        data_key = f'{key}/data'
        data = _TrainsData.from_hdf(path, key=data_key)

        with h5py.File(path, mode='r') as file:
            if data_key not in file:
                raise KeyError(f'HDF5 key {data_key!r} does not exist')

            node = file[data_key]

            if not isinstance(node, h5py.Group):
                raise TypeError(f'Trains payload {data_key!r} must be an HDF5 group')

            support_node = node.get('support')

            if not isinstance(support_node, h5py.Dataset):
                raise TypeError('Trains payload support must be an HDF5 dataset')

            support_values = np.asarray(support_node[...], dtype=float)

        if support_values.shape != (3,):
            raise ValueError('Trains support must contain start, stop, and ref')

        support = nocte.core.windows.Win(
            support_values[0],
            support_values[1],
            ref=support_values[2],
        )

        return cls(data, meta, support)

    # ------------------------------------------------------------------
    # representation and validation

    def to_frame(self) -> pd.DataFrame:
        """Return lightweight per-train summaries together with metadata."""
        counts = self.counts()
        if self.support.length == 0:
            rates = pd.Series(np.nan, index=self.index, name='rate')
        else:
            rates = self.rates()

        summary = pd.concat([counts, rates], axis=1)
        return pd.concat([summary, self.meta], axis=1)

    def _repr_html_(self) -> str:
        return self.to_frame()._repr_html_()  # type: ignore

    def _validate_support(self) -> None:
        start = self.support.time_at('start')
        stop = self.support.time_at('stop')

        for position, times in enumerate(self._data.values):
            if len(times) == 0:
                continue
            if times[0] < start or times[-1] >= stop:
                train_id = self.index[position]
                raise ValueError(
                    f'Train {train_id!r} contains timestamps outside support '
                    f'[{start}, {stop})'
                )

    def _require_within_support(self, win: nocte.core.windows.Win) -> None:
        if not win.contained_in(self.support):
            raise ValueError(f'Window {win} lies outside train support {self.support}')

    def _require_range_within_support(
        self,
        start: float,
        stop: float,
        *,
        name: str,
    ) -> None:
        support_start = self.support.time_at('start')
        support_stop = self.support.time_at('stop')

        if start < support_start or support_stop < stop:
            raise ValueError(
                f'{name} range [{start}, {stop}) lies outside train support '
                f'[{support_start}, {support_stop})'
            )

    @staticmethod
    def _positive_float(value: float, *, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return value
