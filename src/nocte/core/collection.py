from __future__ import annotations

import abc
import collections.abc
import typing

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

How = typing.Literal['all', 'any']

IndexLike = int | collections.abc.Iterable[int] | np.ndarray | pd.Index

MaskLike = collections.abc.Sequence[bool] | np.ndarray | pd.Series

ItemT = typing.TypeVar('ItemT')
PbarT = typing.TypeVar('PbarT')


class Collection(abc.ABC, typing.Generic[ItemT]):
    """Common metadata selection API for nocte collections."""

    meta: pd.DataFrame

    @abc.abstractmethod
    def _sel_pos(self, positions: np.ndarray) -> typing.Self: ...

    @abc.abstractmethod
    def _get_pos(self, position: int) -> ItemT: ...

    @staticmethod
    def _default_meta(n_items: int) -> pd.DataFrame:
        """Return a default metadata DataFrame for a collection."""
        return pd.DataFrame(
            index=pd.RangeIndex(n_items, name='item_id'),
        )

    def __len__(self) -> int:
        """The number of items in the collection."""
        return len(self.meta)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def index(self) -> pd.Index:
        return self.meta.index

    @property
    def columns(self) -> pd.Index:
        return self.meta.columns

    def __getitem__(self, key: typing.Any) -> typing.Any:
        return self.meta[key]

    def __setitem__(self, key: typing.Any, value: typing.Any) -> None:
        self.meta[key] = value

    def _validate_meta(self, n_items: int) -> None:
        """Validate the core metadata invariants of a collection."""
        if len(self.meta) != n_items:
            raise ValueError(
                f'meta has {len(self.meta)} rows, '
                f'but the collection has {n_items} items'
            )

        if isinstance(self.meta.index, pd.MultiIndex):
            raise TypeError('collection index must be single-level')

        if not self.meta.index.is_unique:
            raise ValueError('collection index must be unique')

        if not pd.api.types.is_integer_dtype(self.meta.index.dtype):
            raise ValueError('collection index must contain integers')

        if self.meta.index.hasnans:
            raise ValueError('collection index cannot contain missing values')

    def _positions(self, labels: IndexLike) -> np.ndarray:
        labels = self._as_index(labels)

        if not labels.is_unique:
            raise ValueError('item labels must be unique')

        positions = self.index.get_indexer(labels)

        missing = positions < 0

        if missing.any():
            raise KeyError(f'item labels not found: {labels[missing].tolist()}')

        return positions

    def _align_series(self, values: pd.Series, name: str) -> pd.Series:
        if (
            len(values) != len(self.index)
            or not values.index.is_unique
            or not self.index.difference(values.index).empty
            or not values.index.difference(self.index).empty
        ):
            raise ValueError(f'{name} index must contain exactly the collection index')

        return values.reindex(self.index)

    # ------------------------------------------------------------------
    # item access

    @staticmethod
    def _as_index(
        labels: IndexLike,
    ) -> pd.Index:
        if isinstance(labels, pd.Index):
            return labels

        if isinstance(labels, (int, np.integer)):
            return pd.Index([labels])

        return pd.Index(labels)

    def get(self, item_id: int | None = None) -> ItemT:
        if item_id is None:
            if len(self) != 1:
                raise ValueError(
                    f'get() without an item ID requires exactly one item, got {len(self)}'
                )
            position = 0
        else:
            position = int(self._positions(item_id)[0])

        return self._get_pos(position)

    def items(self) -> collections.abc.Iterator[tuple[int, ItemT]]:
        for position, item_id in enumerate(self.index):
            yield int(item_id), self._get_pos(position)

    # ------------------------------------------------------------------
    # selection

    def sel(
        self,
        rows: IndexLike | None = None,
        /,
        *,
        invert: bool = False,
        **col_values: typing.Any,
    ) -> typing.Self:
        if rows is not None and col_values:
            raise ValueError('provide either item indices or metadata values, not both')

        if rows is not None:
            return self.sel_index(
                rows,
                invert=invert,
            )

        if col_values:
            return self.sel_match(
                invert=invert,
                **col_values,
            )

        raise ValueError('provide item indices or metadata values')

    def sel_index(self, labels: IndexLike, /, *, invert: bool = False) -> typing.Self:
        positions = self._positions(labels)

        if invert:
            mask = np.ones(len(self), dtype=bool)
            mask[positions] = False
            positions = np.flatnonzero(mask)

        return self._sel_pos(positions)

    def sel_mask(
        self,
        mask: MaskLike,
        /,
        *,
        invert: bool = False,
    ) -> typing.Self:
        if isinstance(mask, pd.Series):
            mask = self._align_series(mask, 'mask').to_numpy()

        mask = np.asarray(mask)

        if mask.ndim != 1:
            raise ValueError('mask must be one-dimensional')

        if mask.dtype != bool:
            raise TypeError('mask must be boolean')

        if len(mask) != len(self):
            raise ValueError('mask length does not match collection length')

        if invert:
            mask = ~mask

        return self._sel_pos(np.flatnonzero(mask))

    def is_match(
        self,
        *,
        how: How = 'all',
        invert: bool = False,
        **col_values: typing.Any,
    ) -> pd.Series:
        criteria = []

        for col, value in col_values.items():
            if not pd.api.types.is_scalar(value):
                raise TypeError(
                    f'sel_match expects scalar values, got {type(value).__name__} '
                    f'for {col!r}; use sel_in() for multiple values'
                )

            values = self.meta[col]

            if pd.isna(value):
                criterion = values.isna()
            else:
                criterion = values.eq(value)

            criteria.append(criterion)

        return self._combine_masks(
            criteria,
            how=how,
            invert=invert,
        )

    def sel_match(
        self,
        *,
        how: How = 'all',
        invert: bool = False,
        **col_values: typing.Any,
    ) -> typing.Self:
        return self.sel_mask(
            self.is_match(
                how=how,
                invert=invert,
                **col_values,
            )
        )

    def is_in(
        self,
        *,
        how: How = 'all',
        invert: bool = False,
        **col_values: collections.abc.Iterable[typing.Any],
    ) -> pd.Series:
        criteria = [self.meta[col].isin(values) for col, values in col_values.items()]

        return self._combine_masks(
            criteria,
            how=how,
            invert=invert,
        )

    def sel_in(
        self,
        *,
        how: How = 'all',
        invert: bool = False,
        **col_values: collections.abc.Iterable[typing.Any],
    ) -> typing.Self:
        return self.sel_mask(
            self.is_in(
                how=how,
                invert=invert,
                **col_values,
            )
        )

    def is_between(
        self,
        *,
        how: How = 'all',
        invert: bool = False,
        **col_ranges: tuple[typing.Any, typing.Any],
    ) -> pd.Series:
        criteria = [
            self.meta[col].between(*value_range)
            for col, value_range in col_ranges.items()
        ]

        return self._combine_masks(
            criteria,
            how=how,
            invert=invert,
        )

    def sel_between(
        self,
        *,
        how: How = 'all',
        invert: bool = False,
        **col_ranges: tuple[typing.Any, typing.Any],
    ) -> typing.Self:
        return self.sel_mask(
            self.is_between(
                how=how,
                invert=invert,
                **col_ranges,
            )
        )

    @staticmethod
    def _combine_masks(
        criteria: list[pd.Series],
        *,
        how: How,
        invert: bool,
    ) -> pd.Series:
        if not criteria:
            raise ValueError('at least one selection criterion is required')

        if how == 'all':
            mask = criteria[0].copy()

            for criterion in criteria[1:]:
                mask &= criterion

        elif how == 'any':
            mask = criteria[0].copy()

            for criterion in criteria[1:]:
                mask |= criterion

        else:
            raise ValueError("how must be 'all' or 'any'")

        if invert:
            mask = ~mask

        return mask

    # ------------------------------------------------------------------
    # ordering

    def sort_values(
        self,
        by: str | collections.abc.Sequence[str],
        *,
        ascending: bool = True,
        na_position: typing.Literal['first', 'last'] = 'last',
    ) -> typing.Self:
        meta = self.meta.sort_values(
            by=by,
            ascending=ascending,
            na_position=na_position,
            inplace=False,
            ignore_index=False,
        )

        return self.sel_index(meta.index)

    def sort_index(
        self,
        *,
        ascending: bool | collections.abc.Sequence[bool] = True,
        na_position: typing.Literal['first', 'last'] = 'first',
    ) -> typing.Self:
        meta = self.meta.sort_index(
            ascending=ascending,
            na_position=na_position,
            inplace=False,
            ignore_index=False,
        )

        return self.sel_index(meta.index)

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
    ) -> typing.Self:
        labels = self.meta.sample(
            n=n,
            frac=frac,
            replace=False,
            ignore_index=False,
        ).index

        return self.sel_index(labels)

    def shuffle(self) -> typing.Self:
        return self.sample(
            frac=1,
        )


def optional_pbar(
    iterable: collections.abc.Iterable[PbarT],
    *,
    total: int | None = None,
    pbar: bool
    | str
    | None
    | collections.abc.Callable[..., collections.abc.Iterable[PbarT]] = False,
    desc: str | None = None,
) -> collections.abc.Iterable[PbarT]:

    if pbar is None or pbar is False:
        return iterable

    if isinstance(pbar, str):
        desc = pbar
        pbar = True

    if pbar is True:
        pbar = tqdm

    return pbar(
        iterable,
        total=total,
        desc=desc,
    )
