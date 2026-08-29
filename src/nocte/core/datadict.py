import functools
from typing import Self

import numpy as np
import pandas as pd

import nocte.core.windows
from nocte.core.collection import Collection


class DataDict(Collection):
    """
    A set of data with some metadata for each one.
    Metadata is stored as a pd.DF table (meta) with one row per data.
    Data are stored as a dictionary with each key being the index in meta.
    Data values are of arbitrary type.
    Useful to select different conditions or experiments.
    """

    def __init__(self, meta: pd.DataFrame, data: dict):
        assert np.all(np.isin(meta.index, list(data.keys())))

        super().__init__(meta)
        self.data = data

    def _apply_mask(self, mask):
        """
        Final method for applying a mask to `meta`. Subclasses override this.
        """
        meta = self.meta.loc[mask]
        assert meta.index.is_unique

        return self.__class__(
            meta=meta,
            data={k: self.data[k] for k in meta.index},
        )

    def apply(self, function, pbar=None, **kwargs):
        """Apply the given callable independently to each entry"""

        processed = {k: function(v, **kwargs) for k, v in self.items(pbar=pbar)}

        return self.__class__(self.meta, processed)

    @classmethod
    def from_dict(cls, data: dict, names=None):
        meta = pd.DataFrame.from_records(list(data.keys()), columns=names)
        data_mapped = {uid: data[tuple(k)] for uid, *k in meta.itertuples()}
        return cls(meta, data_mapped)

    @classmethod
    def from_split(
        cls,
        obj: Collection,
        *,
        by: str | list[str],
        sort: bool = False,
    ) -> 'DataDict':
        """
        Split a Collection into a DataDict by one or more columns
        in its registry.

        Each split becomes one object in `data`
        The DataDict registry contains group-level invariant metadata
        """

        meta = obj.meta

        if isinstance(by, str):
            by = [by]

        for col in by:
            if col not in meta.columns:
                raise KeyError(f"Column '{col}' not found in registry")

        data = {}
        reg_rows = []

        gb = meta.groupby(by, sort=sort)

        for new_id, (_, idx) in enumerate(gb.groups.items()):
            subset = obj.sel_mask(idx)

            subreg = meta.loc[idx]

            # find invariant columns within this group
            meta = {}
            for col in subreg.columns:
                vals = subreg[col]
                if vals.nunique(dropna=False) == 1:
                    meta[col] = vals.iloc[0]

            data[new_id] = subset
            reg_rows.append(meta)

        dd_reg = pd.DataFrame(reg_rows)
        dd_reg.index.name = 'id'

        return cls(
            meta=dd_reg,
            data=data,
        )

    def to_hdf(
        self,
        filename,
        key='dd',
        item_key_fmt='{k}',
        pbar=None,
        storer=None,
    ):
        """
        Store data and registry to HDF5.
        Data must implement "to_hdf" or "store_hdf" (e.g. pd.DataFrame)
        """
        self.meta.to_hdf(filename, key=f'{key}_reg')

        index = nocte.core.collection.optional_pbar(
            self.index, total=len(self.index), desc='storing', pbar=pbar
        )

        for k in index:
            v = self.get(k)

            item_key = item_key_fmt.format(k=k)
            item_key = f'{key}_{item_key}'

            if storer is not None:
                storer(v, filename, key=item_key)

            elif hasattr(v, 'to_hdf'):
                v.to_hdf(filename, key=item_key)

            elif hasattr(v, 'store_hdf'):
                v.store_hdf(filename, key=item_key)

            else:
                raise TypeError(
                    f'Object of type {type(v)} does not support '
                    f'HDF5 storage (missing to_hdf or store_hdf)'
                )

    @classmethod
    def from_hdf(
        cls, filename, key='dd', loader=pd.read_hdf, item_key_fmt='{k}', pbar=True
    ):
        """
        Load data and registry from HDF5.
        """
        # noinspection PyTypeChecker
        meta: pd.DataFrame = pd.read_hdf(filename, key=f'{key}_reg')

        data = {}
        index = cls._optional_pbar(
            meta.index, total=len(meta.index), desc='loading', pbar=pbar
        )

        for k in index:
            item_key = item_key_fmt.format(k=k)
            item_key = f'{key}_{item_key}'

            data[k] = loader(filename, key=item_key)

        return cls(meta, data)

    def flatten(
        self,
        *,
        inner_name: str | None = None,
        outer_name: str | None = None,
    ) -> Self:

        data = {}
        reg_rows = []
        new_id = 0

        for outer_key, inner in self.items():
            if not isinstance(inner, DataDict):
                raise TypeError('flatten() requires all items to be DataDicts')

            outer_meta = self.meta.loc[outer_key]

            for inner_key, item in inner.items():
                inner_meta = inner.meta.loc[inner_key]

                meta = {}

                # merge metadata with consistency check
                for k, v in outer_meta.items():
                    meta[k] = v

                for k, v in inner_meta.items():
                    if k in meta and meta[k] != v:
                        raise ValueError(
                            f"Inconsistent metadata for column '{k}': {meta[k]} vs {v}"
                        )
                    meta[k] = v

                if outer_name is not None:
                    if outer_name in meta:
                        raise ValueError(
                            'Key names collide with existing metadata columns'
                        )
                    meta[outer_name] = outer_key

                if inner_name is not None:
                    if inner_name in meta:
                        raise ValueError(
                            'Key names collide with existing metadata columns'
                        )
                    meta[inner_name] = inner_key

                data[new_id] = item
                reg_rows.append(meta)
                new_id += 1

        meta = pd.DataFrame.from_records(reg_rows)
        meta.index.name = 'id'

        return self.__class__(meta=meta, data=data)

    def __len__(self):
        return len(self.meta)

    def replace_data(self, data):
        return self.__class__(
            meta=self.meta.copy(),
            data=data,
        )

    def simplify(self):
        """drop columns with only one value repeated"""
        simpler = self.meta[[c for c, s in self.meta.items() if s.nunique() > 1]]

        return self.__class__(
            simpler,
            self.data,
        )

    def get(self, idx=None):
        """return a single item. If no index it's given, we assume there is only one"""
        if idx is None:
            assert len(self.index) == 1, f'Found too many traces:\n{self.meta}'
            idx = self.index[0]

        return self.data[idx]

    def items(self, col=None, *, pbar=None):
        """
        returns an iterator to go over each data object:
        If 'col' is None, then the key will be the index.
        If it is not, then the key will be the value of the corresponding column.

            for k, data in dd.items('exp_name', pbar=True):
                pass

        """
        # Note we want to respect the order of the registry, not of the dict
        for k in nocte.core.collection.optional_pbar(
            self.index, total=len(self.index), pbar=pbar
        ):
            idx = k if col is None else self.meta.loc[k, col]
            yield idx, self.data[k]

    def sort_values(self, *args, **kwargs):
        meta = self.meta.sort_values(*args, **kwargs)

        return self.__class__(
            meta=meta,
            data={k: self.data[k] for k in meta.index},
        )

    def sort_index(self, *args, **kwargs):
        meta = self.meta.sort_index(*args, **kwargs)

        return self.__class__(
            meta=meta,
            data={k: self.data[k] for k in meta.index},
        )

    def extract_df(self, func, pbar=None, **kwargs) -> pd.DataFrame:
        """
        Extract one pd.Series or pd.DataFrame for each item and
        concatenate them all along axis=1
        """
        results = {
            tuple(key): func(key, data, **kwargs) for key, data in self.items(pbar=pbar)
        }

        df = pd.concat(
            results,
            axis=1,
        )

        names = list(self.meta.columns) + df.columns.names[len(self.meta.columns) :]
        df.rename_axis(columns=names, inplace=True)

        return df

    def iterby(self, by, pbar=None):  # TODO homogenize names
        groups = self.meta.groupby(by).groups.items()

        for k, uids in nocte.core.collection.optional_pbar(
            groups, total=len(groups), desc=str(by), pbar=pbar
        ):
            subset = self.sel_mask(uids)
            subset.meta.drop(by, axis=1, inplace=True)
            yield k, subset

    @classmethod
    def concat_dict(cls, stackset_dict, names):  # TODO homogenize names

        merged_reg = {}

        merged_data = {}

        global_id = 0

        for key, s in stackset_dict.items():
            merged_reg[key] = s.meta.rename_axis(index='local_id')

            for data in s.data.values():
                merged_data[global_id] = data
                global_id += 1

        merged_reg = pd.concat(merged_reg, names=names)

        merged_reg = merged_reg.reset_index().drop(columns='local_id')

        return cls(merged_reg, merged_data)

    @classmethod
    def combine(cls, left, right, func, *, left_ref, right_ref):  # TODO deprecate

        result = cls._combine_idcs(
            left=left,
            right=right,
            left_ref=left_ref,
            right_ref=right_ref,
        )

        return result.apply(
            lambda pair: func(
                left.get(pair[0]),
                right.get(pair[1]),
            )
        )

    @classmethod
    def _combine_idcs(cls, left, right, *, left_ref, right_ref, **merge_kwargs):

        merged = cls.match(
            left,
            right,
            left_ref=left_ref,
            right_ref=right_ref,
            **merge_kwargs,
        )

        indices = merged[[left_ref, right_ref]].values

        results = dict(zip(merged.index.values, indices))

        return cls(merged, results)

    @functools.wraps(pd.DataFrame.reset_index)
    def reset_index(self, *args, drop=True, **kwargs):

        meta = self.meta.reset_index(*args, drop=drop, **kwargs)

        mapping = pd.Series(meta.index, index=self.meta.index)

        data = {mapping[k]: d for k, d in self.data.items()}

        return self.__class__(meta, data)

    @functools.wraps(pd.DataFrame.set_index)
    def set_index(self, *args, **kwargs):

        meta = self.meta.set_index(*args, **kwargs)

        mapping = pd.Series(meta.index, index=self.meta.index)

        data = {mapping[k]: d for k, d in self.data.items()}

        return self.__class__(meta, data)

    def _extract(self, wins: nocte.core.windows.Windows):
        paired = self._combine_idcs(
            self,
            wins,
            left_ref='dd_idx',
            right_ref='win_idx',
        )

        result = {
            k: DataDict._crop_item(
                self.get(dd_idx),
                wins.get(win_idx),
            )
            for k, (dd_idx, win_idx) in paired.items()
        }

        result = self.__class__(
            paired.meta,
            result,
        )

        return result

    def crop(self, win: nocte.core.windows.Win):
        result = {
            k: DataDict._crop_item(
                data,
                win,
            )
            for k, data in self.items()
        }

        result = self.__class__(self.meta, result)

        return result

    @staticmethod
    def _crop_item(item, window: nocte.core.windows.Win):
        if hasattr(item, 'crop'):
            return item.crop(window)

        if isinstance(item, pd.DataFrame):
            idx = item.index
            mask = (idx >= window.start) & (idx <= window.stop)
            return item.loc[mask]

        raise TypeError(f'Object of type {type(item)} cannot be cropped')

    def shift_time(self, ts: pd.Series | np.ndarray | float):
        if isinstance(ts, (int, float, np.ndarray)):
            ts = pd.Series(ts, index=self.index)

        if not ts.index.equals(self.index):
            raise ValueError('shift_time Series must be indexed like DataDict')

        result = {
            k: DataDict._shift_time_item(
                item,
                ts.loc[k],
            )
            for k, item in self.items()
        }

        return self.__class__(self.meta, result)

    @staticmethod
    def _shift_time_item(item, dt):
        if hasattr(item, 'shift_time'):
            return item.shift_time(dt)

        if isinstance(item, pd.DataFrame):
            out = item.copy()
            out.index = out.index + dt
            return out

        raise TypeError(f'Object of type {type(item)} cannot be time-shifted')

    def extract(self, wins: nocte.core.windows.Windows, align=None):

        extracted = self._extract(wins)

        if align is not None:
            refs = wins.relative_time(align)
            shifts: np.ndarray = extracted['win_idx'].map(refs).values
            assert not np.isnan(shifts).any()
            extracted = extracted.shift_time(-1 * shifts)

        return extracted
