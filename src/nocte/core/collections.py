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


class Collection(abc.ABC):
    """Common metadata selection API for nocte collections."""

    meta: pd.DataFrame

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def _take_pos(self, positions: np.ndarray) -> typing.Self: ...

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

    def _validate_meta(self) -> None:
        """Validate the core metadata invariants of a collection."""
        if len(self.meta) != len(self):
            raise ValueError(
                f'meta has {len(self.meta)} rows, but the collection has {len(self)} items'
            )

        if not self.meta.index.is_unique:
            raise ValueError('collection index must be unique')

    def _positions(self, labels: IndexLike) -> np.ndarray:
        labels = self._as_index(labels)
        positions = self.index.get_indexer(labels)

        missing = positions < 0

        if missing.any():
            raise KeyError(f'item labels not found: {labels[missing].tolist()}')

        return positions

    @staticmethod
    def _as_index(
        labels: IndexLike,
    ) -> pd.Index:
        if isinstance(labels, pd.Index):
            return labels

        if isinstance(labels, (int, np.integer)):
            return pd.Index([labels])

        return pd.Index(labels)

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

        return self._take_pos(positions)

    def sel_mask(
        self,
        mask: MaskLike,
        /,
        *,
        invert: bool = False,
    ) -> typing.Self:
        if isinstance(mask, pd.Series):
            if not mask.index.equals(self.index):
                raise ValueError('mask index does not match collection index')

            mask = mask.to_numpy()

        mask = np.asarray(mask)

        if mask.ndim != 1:
            raise ValueError('mask must be one-dimensional')

        if mask.dtype != bool:
            raise TypeError('mask must be boolean')

        if len(mask) != len(self):
            raise ValueError('mask length does not match collection length')

        if invert:
            mask = ~mask

        return self._take_pos(np.flatnonzero(mask))

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
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Self:
        inplace = kwargs.pop('inplace', False)
        ignore_index = kwargs.pop('ignore_index', False)

        if inplace:
            raise TypeError("'inplace=True' is not supported")

        if ignore_index:
            raise TypeError("'ignore_index=True' is not supported")

        meta = self.meta.sort_values(
            *args,
            inplace=False,
            ignore_index=False,
            **kwargs,
        )

        return self.sel_index(meta.index)

    def sort_index(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Self:
        inplace = kwargs.pop('inplace', False)
        ignore_index = kwargs.pop('ignore_index', False)

        if inplace:
            raise TypeError("'inplace=True' is not supported")

        if ignore_index:
            raise TypeError("'ignore_index=True' is not supported")

        meta = self.meta.sort_index(
            *args,
            inplace=False,
            ignore_index=False,
            **kwargs,
        )

        return self.sel_index(meta.index)

    def sample(
        self,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> typing.Self:
        ignore_index = kwargs.pop('ignore_index', False)

        if ignore_index:
            raise TypeError("'ignore_index=True' is not supported")

        if kwargs.get('replace', False):
            raise TypeError("'replace=True' is not supported")

        labels = self.meta.sample(
            *args,
            ignore_index=False,
            **kwargs,
        ).index

        return self.sel_index(labels)

    def shuffle(self) -> typing.Self:
        return self.sample(
            frac=1,
            replace=False,
        )

    @staticmethod
    def _optional_pbar(
        iterable,
        *,
        total: int | None = None,
        pbar: bool
        | str
        | None
        | collections.abc.Callable[..., typing.Iterable] = False,
        desc: str | None = None,
    ):

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
