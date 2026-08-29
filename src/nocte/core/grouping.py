from __future__ import annotations

import collections.abc
import importlib
import pathlib
import typing

import h5py
import numpy as np
import pandas as pd

import nocte.core.collection
import nocte.core.hdf


class CollectionLike(typing.Protocol):
    """Minimal protocol for objects that can be contained in a Grouping."""

    meta: pd.DataFrame

    @property
    def index(self) -> pd.Index: ...

    def __len__(self) -> int: ...

    def sel_index(
        self,
        labels: nocte.core.collection.IndexLike,
        /,
        *,
        invert: bool = False,
    ) -> typing.Self: ...


GroupedT = typing.TypeVar(
    'GroupedT',
    bound=CollectionLike,
)

HDFGroupedT = typing.TypeVar(
    'HDFGroupedT',
    bound=nocte.core.hdf.HDFCollection[typing.Any],
)

MappedT = typing.TypeVar(
    'MappedT',
    bound=CollectionLike,
)

ResultT = typing.TypeVar('ResultT')


def _resolve_type(
    module_name: str,
    qualname: str,
) -> type[typing.Any]:
    module = importlib.import_module(module_name)

    obj: typing.Any = module

    for name in qualname.split('.'):
        obj = getattr(obj, name)

    if not isinstance(obj, type):
        raise TypeError(f'stored grouped type {module_name}.{qualname} is not a class')

    return obj


LoadedT = typing.TypeVar(
    'LoadedT',
    bound=nocte.core.hdf.HDFCollection,
)


class _GroupingData(typing.Generic[GroupedT]):
    """Internal positional storage for homogeneous grouped collection."""

    def __init__(
        self,
        items: collections.abc.Sequence[GroupedT],
    ):
        items = tuple(items)

        if items:
            item_type = type(items[0])

            if any(type(item) is not item_type for item in items):
                raise TypeError('all Grouping items must have the same concrete type')

        self._items: tuple[GroupedT, ...] = items

    def __len__(self) -> int:
        return len(self._items)

    def get_pos(
        self,
        position: int,
    ) -> GroupedT:
        return self._items[position]

    def sel_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        positions = np.asarray(
            positions,
            dtype=np.intp,
        )

        return self.__class__(tuple(self._items[position] for position in positions))

    def copy(self) -> typing.Self:
        return self

    def to_hdf(
        self: _GroupingData[HDFGroupedT],
        path: str | pathlib.Path,
        *,
        key: str,
        pbar: nocte.core.collection.PBarParamT = None,
    ) -> None:
        """
        Store the grouped payload.

        The payload is an HDF5 group containing one complete serialized
        child collection per position. The target key must not already exist.
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
            desc='storing groups',
        )

        for position in positions:
            item = self.get_pos(position)

            item.to_hdf(
                path,
                key=f'{key}/item_{position}',
                overwrite=False,
            )

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        item_type: type[LoadedT],
        pbar: nocte.core.collection.PBarParamT = None,
    ) -> _GroupingData[LoadedT]:
        """
        Load a homogeneous grouped payload.

        ``item_type`` supplies the concrete collection class responsible for
        loading each serialized child.
        """
        key = nocte.core.hdf.normalize_hdf_key(key)

        with h5py.File(path, mode='r') as file:
            if key not in file:
                raise KeyError(f'HDF5 key {key!r} does not exist')

            node = file[key]

            if not isinstance(node, h5py.Group):
                raise TypeError(f'Grouping payload {key!r} must be an HDF5 group')

            names = list(node.keys())

        positions: list[int] = []

        for name in names:
            prefix = 'item_'

            if not name.startswith(prefix):
                raise ValueError(f'unexpected key {name!r} in Grouping payload {key!r}')

            suffix = name[len(prefix) :]

            try:
                position = int(suffix)
            except ValueError as exc:
                raise ValueError(f'invalid Grouping item key {name!r}') from exc

            positions.append(position)

        positions.sort()

        expected = list(range(len(positions)))

        if positions != expected:
            raise ValueError(
                'Grouping payload positions must be contiguous and start at zero'
            )

        iterator = nocte.core.collection.optional_pbar(
            positions,
            total=len(positions),
            pbar=pbar,
            desc='loading groups',
        )

        items: list[LoadedT] = []

        for position in iterator:
            item = item_type.from_hdf(
                path,
                key=f'{key}/item_{position}',
            )

            items.append(item)

        return _GroupingData[LoadedT](items)


class Grouping(
    nocte.core.collection.Collection[GroupedT],
    typing.Generic[GroupedT],
):
    """
    Indexed collection of homogeneous collection-like objects.

    Grouping is the materialized result of grouping another collection.
    Each outer metadata row describes one contained subcollection.

    The outer index identifies groups. Contained collections preserve their
    own item identities.

    Contained objects must satisfy CollectionLike, and all contained objects
    in one Grouping must have the same concrete type.
    """

    def __init__(
        self,
        data: _GroupingData[GroupedT],
        meta: pd.DataFrame,
    ):
        if not isinstance(data, _GroupingData):
            raise TypeError('data must be _GroupingData')

        self._data = data
        self.meta = meta.copy()

        self._validate_meta(len(self._data))

    # ------------------------------------------------------------------
    # construction

    @staticmethod
    def _default_meta(
        n_items: int,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            index=pd.RangeIndex(
                n_items,
                name='group_id',
            ),
        )

    @classmethod
    def from_items(
        cls,
        items: collections.abc.Sequence[GroupedT],
        *,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        data = _GroupingData(items)

        return cls(
            data,
            cls._default_meta(len(data)) if meta is None else meta,
        )

    @classmethod
    def from_groupby(
        cls,
        obj: GroupedT,
        *,
        by: str | list[str],
        sort: bool = False,
    ) -> Grouping[GroupedT]:
        """
        Materialize metadata groups from a collection-like object.

        Each child collection preserves the item identities of the source
        collection. The Grouping receives new outer identities named
        ``group_id``.

        Metadata columns that are invariant within a group are promoted to
        the corresponding outer metadata row.

        Missing values in grouping columns form their own groups rather than
        causing source items to be silently dropped.
        """
        columns = [by] if isinstance(by, str) else list(by)

        if not columns:
            raise ValueError('by must contain at least one metadata column')

        missing = set(columns).difference(obj.meta.columns)

        if missing:
            raise KeyError(f'unknown metadata columns: {sorted(missing)}')

        groups: list[GroupedT] = []
        group_meta: list[dict[typing.Hashable, typing.Any]] = []

        grouped = obj.meta.groupby(
            by,
            sort=sort,
            dropna=False,
            observed=True,
        )

        for labels in grouped.groups.values():
            labels = pd.Index(labels)

            groups.append(obj.sel_index(labels))

            subset_meta = obj.meta.loc[labels]

            invariant = {
                column: values.iloc[0]
                for column, values in subset_meta.items()
                if values.nunique(dropna=False) == 1
            }

            group_meta.append(invariant)

        if group_meta:
            meta = pd.DataFrame.from_records(group_meta)
        else:
            meta = pd.DataFrame(columns=columns)

        meta.index = pd.RangeIndex(
            len(meta),
            name='group_id',
        )

        return cls.from_items(
            groups,
            meta=meta,
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
    ) -> GroupedT:
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
        function: collections.abc.Callable[..., MappedT],
        /,
        *args: typing.Any,
        pbar: nocte.core.collection.PBarParamT = None,
        **kwargs: typing.Any,
    ) -> Grouping[MappedT]:
        """
        Apply a one-to-one collection transformation to every group.

        The result for each group may have a different collection-like type
        from the input, but all results must have the same concrete type.

        Outer group identities and metadata are preserved.
        """
        iterator = (group for _, group in self.items())

        iterator = nocte.core.collection.optional_pbar(
            iterator,
            total=len(self),
            pbar=pbar,
        )

        results = [
            function(
                group,
                *args,
                **kwargs,
            )
            for group in iterator
        ]

        return Grouping(
            _GroupingData(results),
            self.meta,
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
        Apply a function independently to every group.

        Results may be arbitrary Python objects and are returned as a Series
        indexed by the outer group IDs.
        """
        iterator = (group for _, group in self.items())

        iterator = nocte.core.collection.optional_pbar(
            iterator,
            total=len(self),
            pbar=pbar,
        )

        results = [
            function(
                group,
                *args,
                **kwargs,
            )
            for group in iterator
        ]

        return pd.Series(
            results,
            index=self.index.copy(),
        )

    # ------------------------------------------------------------------
    # serialization

    def to_hdf(
        self: Grouping[HDFGroupedT],
        path: str | pathlib.Path,
        *,
        key: str = 'grouping',
        overwrite: bool = False,
        pbar: nocte.core.collection.PBarParamT = False,
    ) -> None:
        """
        Store this Grouping in HDF5.

        Layout
        ------
        /<key>
            attrs:
                kind = 'grouping'
                nocte_version = <current version>

            /meta
                outer group metadata

            /data
                /item_0
                /item_1
                ...

        Each item under ``data`` is a complete serialized child collection.

        If ``overwrite`` is False, an existing collection root raises
        FileExistsError. If True, the complete existing subtree is removed
        before writing.
        """
        key = nocte.core.hdf.prepare_hdf_key(
            path,
            key,
            overwrite=overwrite,
        )

        self.meta.to_hdf(
            path,
            key=f'{key}/meta',
            mode='a',
            format='fixed',
        )

        nocte.core.hdf.write_hdf_collection_attrs(
            path,
            key,
            kind='grouping',
        )

        self._data.to_hdf(
            path,
            key=f'{key}/data',
            pbar=pbar,
        )

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        item_type: type[HDFGroupedT],
        key: str = 'grouping',
        pbar: nocte.core.collection.PBarParamT = False,
    ) -> Grouping[HDFGroupedT]:
        key = nocte.core.hdf.check_hdf_collection_attrs(
            path,
            key,
            expected_kind='grouping',
        )

        meta = pd.read_hdf(
            path,
            key=f'{key}/meta',
        )

        if not isinstance(meta, pd.DataFrame):
            raise TypeError(f'HDF metadata at {key!r}/meta is not a DataFrame')

        data = _GroupingData.from_hdf(
            path,
            key=f'{key}/data',
            item_type=item_type,
            pbar=pbar,
        )

        return Grouping[HDFGroupedT](
            data=data,
            meta=meta,
        )
