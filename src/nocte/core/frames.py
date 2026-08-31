from __future__ import annotations

import collections.abc
import pathlib
import typing

import h5py
import numpy as np
import pandas as pd

import nocte.core.collection
import nocte.core.hdf

ResultT = typing.TypeVar('ResultT')


class _FramesData:
    """Internal positional storage for a collection of pandas DataFrames."""

    def __init__(
        self,
        frames: collections.abc.Sequence[pd.DataFrame],
    ):
        frames = tuple(frames)

        if any(
            not isinstance(
                frame,
                pd.DataFrame,
            )
            for frame in frames
        ):
            raise TypeError('all Frames items must be pandas DataFrames')

        self._frames = tuple(frame.copy(deep=True) for frame in frames)

    @classmethod
    def _from_owned(
        cls,
        frames: tuple[pd.DataFrame, ...],
    ) -> typing.Self:
        result = cls.__new__(cls)
        result._frames = frames

        return result

    def __len__(self) -> int:
        return len(self._frames)

    def get_pos(
        self,
        position: int,
    ) -> pd.DataFrame:
        return self._frames[position].copy(deep=True)

    def get_stored_pos(
        self,
        position: int,
    ) -> pd.DataFrame:
        return self._frames[position]

    def sel_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        positions = np.asarray(
            positions,
            dtype=np.intp,
        )

        return self._from_owned(tuple(self._frames[position] for position in positions))

    def copy(self) -> typing.Self:
        return self

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
        pbar: nocte.core.collection.PBarParamT = None,
    ) -> None:
        """
        Store the DataFrame payload.

        The payload is an HDF5 group containing one pandas DataFrame per
        positional item. The target key must not already exist.
        """
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='a') as file:
            if key in file:
                raise FileExistsError(f'HDF5 key {key!r} already exists')

            file.create_group(key)

        positions = nocte.core.collection.optional_pbar(
            range(len(self)),
            total=len(self),
            pbar=pbar,
            desc='storing frames',
        )

        for position in positions:
            self._frames[position].to_hdf(
                path,
                key=f'{key}/item_{position}',
                mode='a',
                format='fixed',
            )

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        pbar: nocte.core.collection.PBarParamT = None,
    ) -> typing.Self:
        """
        Load a DataFrame payload previously stored with to_hdf().

        Stored item positions must be contiguous and start at zero.
        """
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='r') as file:
            if key not in file:
                raise KeyError(f'HDF5 key {key!r} does not exist')

            node = file[key]

            if not isinstance(node, h5py.Group):
                raise TypeError(f'Frames payload {key!r} must be an HDF5 group')

            names = list(node.keys())

        positions: list[int] = []

        for name in names:
            prefix = 'item_'

            if not name.startswith(prefix):
                raise ValueError(f'unexpected key {name!r} in Frames payload {key!r}')

            suffix = name[len(prefix) :]

            try:
                position = int(suffix)
            except ValueError as exc:
                raise ValueError(f'invalid Frames item key {name!r}') from exc

            positions.append(position)

        positions.sort()

        expected = list(range(len(positions)))

        if positions != expected:
            raise ValueError(
                'Frames payload positions must be contiguous and start at zero'
            )

        iterator = nocte.core.collection.optional_pbar(
            positions,
            total=len(positions),
            pbar=pbar,
            desc='loading frames',
        )

        frames: list[pd.DataFrame] = []

        for position in iterator:
            loaded = pd.read_hdf(
                path,
                key=f'{key}/item_{position}',
            )

            if not isinstance(loaded, pd.DataFrame):
                raise TypeError(f'stored Frames item {position} must be a DataFrame')

            frames.append(loaded)

        return cls(frames)


class Frames(
    nocte.core.hdf.HDFCollection[pd.DataFrame],
):
    """
    Indexed collection with one pandas DataFrame per item.

    Each row of meta describes one stored DataFrame. Payload alignment is
    positional while meta.index provides the stable public item identity.

    DataFrames may have different shapes, indices, columns, and dtypes.

    Frames has no temporal semantics. DataFrame-to-DataFrame transformations
    use map(); arbitrary per-item transformations use apply().
    """

    def __init__(
        self,
        data: _FramesData,
        meta: pd.DataFrame,
    ):
        if not isinstance(
            data,
            _FramesData,
        ):
            raise TypeError('data must be _FramesData')

        self._data = data
        self.meta = meta.copy()
        self._validate_meta(len(self._data))

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def from_items(
        cls,
        frames: collections.abc.Sequence[pd.DataFrame],
        *,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        data = _FramesData(frames)

        return cls(
            data,
            cls._default_meta(len(data), name='frame') if meta is None else meta,
        )

    # ------------------------------------------------------------------
    # collection primitives

    def _sel_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        return self.__class__(
            self._data.sel_pos(positions),
            self.meta.iloc[positions],
        )

    def _get_pos(
        self,
        position: int,
    ) -> pd.DataFrame:
        return self._data.get_pos(position)

    def copy(self) -> typing.Self:
        return self.__class__(
            self._data.copy(),
            self.meta.copy(),
        )

    # ------------------------------------------------------------------
    # transformation

    def map(
        self,
        function: collections.abc.Callable[..., pd.DataFrame],
        /,
        *args: typing.Any,
        pbar: nocte.core.collection.PBarParamT = None,
        **kwargs: typing.Any,
    ) -> typing.Self:
        """
        Apply a one-to-one DataFrame transformation to every item.

        The function must return a DataFrame for every input item. Item
        identities and metadata are preserved.
        """
        iterator = (frame for _, frame in self.items())

        iterator = nocte.core.collection.optional_pbar(
            iterator,
            total=len(self),
            pbar=pbar,
        )

        frames = [
            function(
                frame,
                *args,
                **kwargs,
            )
            for frame in iterator
        ]

        return self.__class__.from_items(
            frames,
            meta=self.meta,
        )

    def apply(
        self,
        function: collections.abc.Callable[..., ResultT],
        /,
        *args: typing.Any,
        pbar: nocte.core.collection.PBarParamT = None,
        **kwargs: typing.Any,
    ) -> pd.Series:
        """
                Apply a function independently to every DataFrame.
        git
                Results may be arbitrary Python objects and are returned as a Series
                indexed by the Frames item index.
        """
        iterator = (frame for _, frame in self.items())

        iterator = nocte.core.collection.optional_pbar(
            iterator,
            total=len(self),
            pbar=pbar,
        )

        results = [
            function(
                frame,
                *args,
                **kwargs,
            )
            for frame in iterator
        ]

        return pd.Series(
            results,
            index=self.index.copy(),
        )

    # ------------------------------------------------------------------
    # serialization

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

        data = _FramesData.from_hdf(path, key=f'{key}/data')

        return cls(data=data, meta=meta)
