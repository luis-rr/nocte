from __future__ import annotations

import dataclasses
import pathlib
import typing

import numpy as np
import pandas as pd

from nocte._coll.events import Events, EventsGrouping
from nocte._coll.sources import Sources, SourcesGrouping
from nocte._coll.windows import Windows, WindowsGrouping
from nocte._core.collection import is_valid_name
from nocte._core.grouping import Grouping
from nocte._core.hdf import HDFCollection

GroupingT = typing.TypeVar(
    'GroupingT',
    bound=Grouping[typing.Any],
)


@dataclasses.dataclass(frozen=True, slots=True)
class _RegistryData:
    """Registry resources grouped by type."""

    wins: WindowsGrouping
    events: EventsGrouping
    sources: SourcesGrouping


RegistryEntry = _RegistryData


class Registry(
    HDFCollection[RegistryEntry],
):
    """
    Indexed metadata registry with grouped temporal and filesystem resources.

    Each resource group has exactly two outer metadata columns:

    - the Registry item identity
    - ``tag``
    """

    def __init__(
        self,
        data: _RegistryData,
        meta: pd.DataFrame,
    ):
        if not isinstance(
            data,
            _RegistryData,
        ):
            raise TypeError('data must be _RegistryData')

        self._data = data
        self.meta = meta.copy()

        self._validate_meta(
            len(self.meta),
        )
        self._validate_data()

    @classmethod
    def from_data(
        cls,
        *,
        meta: pd.DataFrame,
        wins: WindowsGrouping,
        events: EventsGrouping,
        sources: SourcesGrouping,
    ) -> typing.Self:
        return cls(
            _RegistryData(
                wins=wins,
                events=events,
                sources=sources,
            ),
            meta,
        )

    # ------------------------------------------------------------------
    # payload

    @property
    def wins(
        self,
    ) -> WindowsGrouping:
        return self._data.wins

    @property
    def events(
        self,
    ) -> EventsGrouping:
        return self._data.events

    @property
    def sources(
        self,
    ) -> SourcesGrouping:
        return self._data.sources

    # ------------------------------------------------------------------
    # collection interface
    def _entry_values(
        self,
        grouping: Grouping[typing.Any],
    ) -> np.ndarray:
        column = grouping.meta[self.name]

        if not isinstance(column, pd.Series):
            raise TypeError(f'grouping metadata column {self.name!r} must be a Series')

        return column.to_numpy()

    def _sel_entry(
        self,
        grouping: GroupingT,
        entry_id: int,
    ) -> GroupingT:
        mask = self._entry_values(grouping) == entry_id
        return grouping.sel_mask(np.asarray(mask, dtype=bool))

    def _sel_entries(
        self,
        grouping: GroupingT,
        entry_ids: np.ndarray,
    ) -> GroupingT:
        mask = np.isin(
            self._entry_values(grouping),
            entry_ids,
        )

        return grouping.sel_mask(np.asarray(mask, dtype=bool))

    def _get_pos(
        self,
        position: int,
    ) -> RegistryEntry:
        entry_ids = np.asarray(
            self.index,
            dtype=np.int64,
        )

        entry_id = int(entry_ids[position])

        return _RegistryData(
            wins=self._sel_entry(
                self.wins,
                entry_id,
            ),
            events=self._sel_entry(
                self.events,
                entry_id,
            ),
            sources=self._sel_entry(
                self.sources,
                entry_id,
            ),
        )

    def _sel_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        positions = np.asarray(
            positions,
            dtype=np.intp,
        )

        all_entry_ids = np.asarray(
            self.index,
            dtype=np.int64,
        )
        entry_ids = all_entry_ids[positions]

        return self.__class__(
            _RegistryData(
                wins=self._sel_entries(
                    self.wins,
                    entry_ids,
                ),
                events=self._sel_entries(
                    self.events,
                    entry_ids,
                ),
                sources=self._sel_entries(
                    self.sources,
                    entry_ids,
                ),
            ),
            self.meta.iloc[positions],
        )

    def copy(
        self,
    ) -> typing.Self:
        return self._sel_pos(
            np.arange(
                len(self),
                dtype=np.intp,
            )
        )

    def rename(
        self,
        name: str,
    ) -> typing.Self:
        """
        Return a Registry with a new item identity namespace.

        The Registry identity is also the foreign-key column used by all
        resource Groupings, so it is renamed there as well.
        """
        if not is_valid_name(name):
            raise ValueError('name must be a non-empty string')

        old_name = self.name

        renamed = self.copy()
        renamed.meta.index = renamed.meta.index.rename(name)

        for grouping in (
            renamed.wins,
            renamed.events,
            renamed.sources,
        ):
            grouping.meta.rename(
                columns={
                    old_name: name,
                },
                inplace=True,
            )

        renamed._validate_data()

        return renamed

    # ------------------------------------------------------------------
    # validation

    def _validate_data(
        self,
    ) -> None:
        self._validate_grouping(
            self.wins,
            expected_type=(WindowsGrouping),
            name='wins',
        )

        self._validate_grouping(
            self.events,
            expected_type=(EventsGrouping),
            name='events',
        )

        self._validate_grouping(
            self.sources,
            expected_type=SourcesGrouping,
            name='sources',
        )

    def _validate_grouping(
        self,
        grouping: Grouping[typing.Any],
        *,
        expected_type: type[Grouping[typing.Any]],
        name: str,
    ) -> None:
        if not isinstance(
            grouping,
            expected_type,
        ):
            raise TypeError(f'{name} must be {expected_type.__name__}')

        expected_columns = {
            self.name,
            'tag',
        }

        if set(grouping.meta.columns) != expected_columns:
            raise ValueError(
                f'{name}.meta columns must '
                'be exactly '
                f'{sorted(expected_columns)}, '
                f'got '
                f'{list(grouping.meta.columns)}'
            )

        entries = grouping.meta[self.name]
        if not isinstance(entries, pd.Series):
            raise TypeError(f'{name}.{self.name} must be a Series')

        if np.asarray(entries.isna(), dtype=bool).any():
            raise ValueError(f'{name}.{self.name} cannot contain missing Registry IDs')

        if len(entries) and not pd.api.types.is_integer_dtype(entries.dtype):
            raise TypeError(f'{name}.{self.name} must contain integer Registry IDs')

        entry_ids = pd.Index(entries.unique())

        missing = entry_ids.difference(self.index)

        if not missing.empty:
            raise ValueError(
                f'{name} references unknown Registry IDs: {missing.tolist()}'
            )

        invalid_tags = [
            tag
            for tag in grouping.meta['tag']
            if (
                not isinstance(
                    tag,
                    tuple,
                )
                or not tag
                or any(
                    not isinstance(
                        part,
                        str,
                    )
                    or not part
                    for part in tag
                )
            )
        ]

        if invalid_tags:
            raise ValueError(
                f'{name}.tag must contain non-empty tuples of non-empty strings'
            )

        duplicated = grouping.meta.duplicated(
            subset=[
                self.name,
                'tag',
            ]
        )

        if duplicated.to_numpy(dtype=bool).any():
            duplicates = grouping.meta.loc[
                duplicated,
                [
                    self.name,
                    'tag',
                ],
            ]

            raise ValueError(
                f'{name} contains duplicate Registry ID/tag pairs:\n{duplicates}'
            )

    # ------------------------------------------------------------------
    # serialization

    def _to_hdf_data(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        self.wins.to_hdf(
            path,
            key=f'{key}/data/wins',
        )

        self.events.to_hdf(
            path,
            key=f'{key}/data/events',
        )

        self.sources.to_hdf(
            path,
            key=f'{key}/data/sources',
        )

    @classmethod
    def _from_hdf_data(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        meta: pd.DataFrame,
    ) -> typing.Self:
        wins = Grouping.from_hdf(
            path,
            key=f'{key}/wins',
            item_type=(Windows),
        )

        events = Grouping.from_hdf(
            path,
            key=f'{key}/events',
            item_type=(Events),
        )

        sources = Grouping.from_hdf(
            path,
            key=f'{key}/data/sources',
            item_type=Sources,
        )

        return cls(
            _RegistryData(
                wins=(
                    WindowsGrouping.from_items(
                        tuple(item for _, item in wins.items()),
                        meta=wins.meta,
                    )
                ),
                events=(
                    EventsGrouping.from_items(
                        tuple(item for _, item in events.items()),
                        meta=events.meta,
                    )
                ),
                sources=(
                    SourcesGrouping.from_items(
                        tuple(item for _, item in sources.items()),
                        meta=sources.meta,
                    )
                ),
            ),
            meta,
        )
