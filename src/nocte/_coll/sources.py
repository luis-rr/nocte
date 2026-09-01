from __future__ import annotations

import collections.abc
import dataclasses
import pathlib
import typing

import numpy as np
import pandas as pd

import nocte._coll.events
import nocte._coll.windows
import nocte._core.collection
import nocte._core.grouping
import nocte._core.hdf

SourceKind = typing.Literal['file', 'dir', 'any']


@dataclasses.dataclass(frozen=True, slots=True)
class Source:
    """Concrete filesystem resource attached to a Registry entry."""

    path: pathlib.Path
    extra: dict[str, object] = dataclasses.field(default_factory=dict)
    kind: SourceKind = 'any'

    def __post_init__(self) -> None:
        if self.kind not in {'file', 'dir', 'any'}:
            raise ValueError("kind must be 'file', 'dir', or 'any'")

        object.__setattr__(self, 'path', pathlib.Path(self.path))
        object.__setattr__(self, 'extra', dict(self.extra))


class Sources(nocte._core.hdf.HDFCollection[Source]):
    """Indexed collection of concrete filesystem Sources."""

    def __init__(
        self,
        data: collections.abc.Sequence[Source],
        meta: pd.DataFrame,
    ):
        data = tuple(data)

        if any(not isinstance(source, Source) for source in data):
            raise TypeError('all Sources payload items must be Source objects')

        self._data: tuple[Source, ...] = data
        self.meta = meta.copy()
        self._validate_meta(len(self._data))

    @classmethod
    def from_sources(
        cls,
        sources: collections.abc.Sequence[Source],
        *,
        meta: pd.DataFrame | None = None,
    ) -> typing.Self:
        sources = tuple(sources)

        if meta is None:
            meta = cls._default_meta(
                len(sources),
                name='source',
            )

        return cls(
            sources,
            meta,
        )

    def _sel_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        positions = np.asarray(
            positions,
            dtype=np.intp,
        )

        return self.__class__(
            tuple(self._data[position] for position in positions),
            self.meta.iloc[positions],
        )

    def _get_pos(
        self,
        position: int,
    ) -> Source:
        return self._data[position]

    def copy(self) -> typing.Self:
        return self.__class__(
            self._data,
            self.meta,
        )

    def groupby(
        self,
        by: str | list[str],
        *,
        sort: bool = False,
    ) -> SourcesGrouping:
        return SourcesGrouping.from_groupby(
            self,
            by=by,
            sort=sort,
        )

    # ------------------------------------------------------------------
    # serialization

    def _to_hdf_data(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        payload = pd.DataFrame(
            {
                'path': [str(source.path) for source in self._data],
                'kind': [source.kind for source in self._data],
                'extra': [source.extra for source in self._data],
            }
        )

        payload.to_hdf(
            path,
            key=f'{key}/data',
            mode='a',
            format='fixed',
        )

    @classmethod
    def _from_hdf_data(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        meta: pd.DataFrame,
    ) -> typing.Self:
        payload = pd.read_hdf(
            path,
            key=f'{key}/data',
        )

        if not isinstance(payload, pd.DataFrame):
            raise TypeError(f'HDF Sources payload at {key!r}/data is not a DataFrame')

        expected_columns = {
            'path',
            'kind',
            'extra',
        }

        if set(payload.columns) != expected_columns:
            raise ValueError(
                'Sources payload columns must be exactly '
                f'{sorted(expected_columns)}, '
                f'got {list(payload.columns)}'
            )

        if len(payload) != len(meta):
            raise ValueError(
                f'Sources payload has {len(payload)} items, '
                f'but metadata has {len(meta)} rows'
            )

        sources: list[Source] = []

        for position in range(len(payload)):
            path_value = payload['path'].iloc[position]
            kind_value = payload['kind'].iloc[position]
            extra_value = payload['extra'].iloc[position]

            if not isinstance(path_value, str):
                raise TypeError('serialized Source.path must be a string')

            if kind_value not in {'file', 'dir', 'any'}:
                raise ValueError(f'invalid serialized Source.kind: {kind_value!r}')

            if not isinstance(extra_value, dict):
                raise TypeError('serialized Source.extra must be a dict')

            sources.append(
                Source(
                    path=pathlib.Path(path_value),
                    kind=typing.cast(SourceKind, kind_value),
                    extra=typing.cast(dict[str, object], extra_value),
                )
            )

        return cls(
            sources,
            meta,
        )


class SourcesGrouping(
    nocte._core.grouping.Grouping[Sources],
):
    """Homogeneous grouping of Sources collections."""
