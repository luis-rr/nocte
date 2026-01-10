import logging
from typing import Self

import functools

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class DataFrameWrapper:
    """
    Base class for objects that wrap around a pandas DataFrames to keep metadata..
    Provides common selection methods like `sel()`.

    All selection converges to "_apply_mask", so subclasses can override that method.
    This class assumes "inplace" are not used as keyword arguments for simplicity.
    """

    def __init__(self, reg: pd.DataFrame):
        self.reg = reg

    @property
    def index(self) -> pd.Index:
        """pd.DataFrame accessor"""
        return self.reg.index

    @property
    def columns(self) -> pd.Index:
        """pd.DataFrame accessor"""
        return self.reg.columns

    @functools.wraps(pd.DataFrame.__getitem__)
    def __getitem__(self, *args, **kwargs):
        return self.reg.__getitem__(*args, **kwargs)

    @functools.wraps(pd.DataFrame.__setitem__)
    def __setitem__(self, *args, **kwargs):
        return self.reg.__setitem__(*args, **kwargs)

    @functools.wraps(pd.DataFrame.__str__)
    def __str__(self):
        return self.reg.__str__()

    @functools.wraps(pd.DataFrame.__repr__)
    def __repr__(self):
        return self.reg.__repr__()

    @property
    def loc(self):
        return self.reg.loc

    @property
    def iloc(self):
        return self.reg.iloc

    # noinspection PyProtectedMember,PyTypeChecker
    @functools.wraps(pd.DataFrame._repr_html_)
    def _repr_html_(self):
        # noinspection PyCallingNonCallable
        return self.reg._repr_html_()

    @property
    def empty(self) -> bool:
        """pd.DataFrame accessor"""
        return self.reg.empty

    @functools.wraps(pd.DataFrame.value_counts)
    def value_counts(self, *args, **kwargs):
        return self.reg.value_counts(*args, **kwargs)

    @functools.wraps(pd.DataFrame.sort_values)
    def sort_values(self, *args, **kwargs):
        return self.__class__(self.reg.sort_values(*args, **kwargs))

    @functools.wraps(pd.DataFrame.sort_index)
    def sort_index(self, *args, **kwargs):
        return self.__class__(self.reg.sort_index(*args, **kwargs))

    def sel(self, rows=None, /, invert=False, **kwargs) -> Self:
        """
        Select rows either by index or by matching column values.
        """
        if rows is not None and kwargs:
            raise ValueError("Provide either row indices or keyword arguments for filtering, not both.")

        if rows is not None:
            if not isinstance(rows, (slice, tuple, list, np.ndarray, pd.Index)):
                rows = [rows]

            return self.sel_mask(rows, invert=invert)

        if kwargs:
            return self.sel_match(**kwargs, invert=invert)

        raise ValueError("Must provide either row indices or keyword arguments.")

    def _apply_mask(self, mask) -> Self:
        """
        Final method for applying an index mask to `reg`. Subclasses override this.
        """
        return self.__class__(
            self.reg.loc[mask]
        )

    def _replace_reg(self, reg) -> Self:
        """
        Method for replacing the reg without masking the index. Subclasses override this.
        """
        return self.__class__(
            reg
        )

    @staticmethod
    def _masks(criterias, *, how='all', invert=False):
        """
        Combine multiple boolean masks
        """
        assert how in ('all', 'any')

        if how == 'all':
            mask = np.all(criterias, axis=0)
        else:
            mask = np.any(criterias, axis=0)

        if invert:
            mask = ~mask

        return mask

    def sel_mask(self, mask, /, invert=False) -> Self:
        """
        Select using a boolean mask
        """
        if invert:
            mask = ~mask

        return self._apply_mask(mask)

    def is_match(self, *, how='all', invert=False, **col_values) -> pd.Series:
        """
        Return a mask of direct comparison of some column.

        We allow to select for missing values by using np.nan.

        For example:
            wins.match(cat='baseline')
        """
        criterias = [
            (self.reg[col] == value) if not pd.isna(value) else self.reg[col].isna()
            for col, value in col_values.items()
        ]

        return self._masks(criterias, how=how, invert=invert)

    def sel_match(self, *, how='all', invert=False, **col_values) -> Self:
        """
        Select by direct comparison of some column.

        We allow to select for missing values by using np.nan.

        For example:
            wins.sel(cat='baseline')
        """
        mask = self.is_match(how=how, invert=invert, **col_values)
        return self.sel_mask(mask)

    def is_between(self, *, how='all', invert=False, **col_ranges) -> pd.Series:
        """
        Return a mask that checks that some columns are within the given range.
        For example:
            wins.between(duration=(0, 60_000))
        """
        criterias = [
            self.reg[col].between(*vrange)
            for col, vrange in col_ranges.items()
        ]

        return self._masks(criterias, how=how, invert=invert)

    def sel_between(self, *, invert=False, how='all', **col_ranges) -> Self:
        """
        Select by direct comparison of some column where values in a range are acceptable.
        For example:
            wins.sel_between(duration=(0, 60_000))
        """
        mask = self.is_between(how=how, invert=invert, **col_ranges)
        return self.sel_mask(mask)

    def is_in(self, *, how='all', invert=False, **col_values) -> pd.Series:
        """
        Return a mask of direct comparison of some column where any of the values are acceptable.
        For example:
            wins.isin(cat=['sws', 'rem'])
        """
        criterias = [
            self.reg[col].isin(values)
            for col, values in col_values.items()
        ]

        return self._masks(criterias, how=how, invert=invert)

    def sel_in(self, *, invert=False, how='all', **col_values) -> Self:
        """
        Select by direct comparison of some column where any of the values are acceptable.
        For example:
            wins.sel_in(cat=['sws', 'rem'])
        """
        mask = self.is_in(invert=invert, how=how, **col_values)
        return self.sel_mask(mask)

    @functools.wraps(pd.DataFrame.sample)
    def sample(self, *args, **kwargs):
        mask = self.reg.sample(*args, **kwargs).index
        return self._apply_mask(mask)

    def shuffle(self):
        """Return a shuffled version of reg."""
        return self.sample(frac=1, replace=False)

    @classmethod
    def match(
            cls,
            left,
            right,
            *,
            left_ref: str,
            right_ref: str,
            how='inner',
            on=None,
            **merge_kwargs,
    ) -> pd.DataFrame:
        """
        Produce a merged registry matching left and right registries.

        The returned DataFrame index uniquely identifies each match.
        Columns contain references into the left and right registries.
        """

        left_reg = left.reg.copy()
        if left_ref in left_reg.columns:
            logging.warning(f'Overriding existing col "{left_ref}" on left reg.')
            left_reg = left_reg.drop(left_ref, axis=1)
        left_reg = left_reg.rename_axis(index=left_ref).reset_index()

        right_reg = right.reg.copy()
        if right_ref in right_reg.columns:
            logging.warning(f'Overriding existing col "{right_ref}" on right reg.')
            right_reg = right_reg.drop(right_ref, axis=1)
        right_reg = right_reg.rename_axis(index=right_ref).reset_index()

        if on is not None:
            merge_kwargs = dict(left_on=on, right_on=on, **merge_kwargs)

        # noinspection PyTypeChecker
        matched = pd.merge(left_reg, right_reg, how=how, **merge_kwargs)

        return matched


def _optional_pbar(iterator, total, pbar, desc=None, many=100):
    """sensible defaults for iterating with an optional progress bar"""

    if isinstance(pbar, str):
        desc = pbar
        pbar = True

    if pbar is None:
        pbar = total > many

    if pbar is True:
        pbar = tqdm

    if pbar is not False:
        return pbar(iterator, total=total, desc=desc)

    else:
        return iterator
