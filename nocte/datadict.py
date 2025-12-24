from tqdm.auto import tqdm

import h5py
import numpy as np
import pandas as pd

from nocte.df_wrapper import DataFrameWrapper, _optional_pbar


class DataDict(DataFrameWrapper):
    """
    A set of data with some metadata for each one.
    Metadata is stored as a pd.DF table (reg) with one row per data.
    Data are stored as a dictionary with each key being the index in reg.
    Data values are of arbitrary type.
    Useful to select different conditions or experiments.
    """

    def __init__(self, reg: pd.DataFrame, data: dict):
        assert np.all(np.isin(reg.index, list(data.keys())))

        super().__init__(reg)
        self.data = data

    def _apply_mask(self, mask):
        """
        Final method for applying a mask to `reg`. Subclasses override this.
        """
        reg = self.reg.loc[mask]
        assert reg.index.is_unique

        return self.__class__(
            reg=reg,
            data={k: self.data[k] for k in reg.index},
        )

    def apply(self, function, pbar=None, **kwargs):
        """Apply the given callable independently to each entry"""

        processed = {
            k: function(v, **kwargs)
            for k, v in self.items(pbar=pbar)
        }

        return self.__class__(self.reg, processed)

    @classmethod
    def from_dict(cls, data: dict, names=None):
        reg = pd.DataFrame.from_records(list(data.keys()), columns=names)
        data_mapped = {uid: data[tuple(k)] for uid, *k in reg.itertuples()}
        return cls(reg, data_mapped)

    def to_hdf(self, filename, key='dd', item_key_fmt='{k}', pbar=None):
        """
        Store data and registry to HDF5.
        Data must implement "to_hdf" or "store_hdf" (e.g. pd.DataFrame)
        """
        self.reg.to_hdf(filename, key=f'{key}_reg')

        index = _optional_pbar(self.index, total=len(self.index), desc='storing', pbar=pbar)

        for k in index:

            v = self.get(k)

            item_key = item_key_fmt.format(k=k)
            item_key = f'{key}_{item_key}'

            if hasattr(v, 'to_hdf'):
                v.to_hdf(filename, key=item_key)

            elif hasattr(v, 'store_hdf'):
                v.store_hdf(filename, key=item_key)

            else:
                raise TypeError(
                    f'Object of type {type(v)} does not support '
                    f'HDF5 storage (missing to_hdf or store_hdf)'
                )

    @classmethod
    def from_hdf(cls, filename, key='dd', loader=pd.read_hdf, item_key_fmt='{k}', pbar=True):
        """
        Load data and registry from HDF5.
        """
        # noinspection PyTypeChecker
        reg: pd.DataFrame = pd.read_hdf(filename, key=f'{key}_reg')

        data = {}
        index = _optional_pbar(reg.index, total=len(reg.index), desc='loading', pbar=pbar)

        for k in index:
            item_key = item_key_fmt.format(k=k)
            item_key = f'{key}_{item_key}'

            data[k] = loader(filename, key=item_key)

        return cls(reg, data)

    def __len__(self):
        return len(self.reg)

    def simplify(self):
        """drop columns with only one value repeated"""
        simpler = self.reg[[c for c, s in self.reg.items() if s.nunique() > 1]]

        return self.__class__(
            simpler,
            self.data,
        )

    def get(self, idx=None):
        """return a single item. If no index it's given, we assume there is only one"""
        if idx is None:
            assert len(self.index) == 1, f'Found too many traces:\n{self.reg}'
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
        for k in _optional_pbar(self.index, total=len(self.index), pbar=pbar):
            idx = k if col is None else self.loc[k, col]
            yield idx, self.data[k]

    def sort_values(self, *args, **kwargs):
        reg = self.reg.sort_values(*args, **kwargs)

        return self.__class__(
            reg=reg,
            data={k: self.data[k] for k in reg.index},
        )

    def sort_index(self, *args, **kwargs):
        reg = self.reg.sort_index(*args, **kwargs)

        return self.__class__(
            reg=reg,
            data={k: self.data[k] for k in reg.index},
        )

    def extract_df(self, func, pbar=None, **kwargs) -> pd.DataFrame:
        """
        Extract one pd.Series or pd.DataFrame for each item and
        concatenate them all along axis=1
        """
        results = {
            tuple(key): func(key, data, **kwargs)
            for key, data in self.items(pbar=pbar)
        }

        df = pd.concat(
            results,
            axis=1,
        )

        names = list(self.reg.columns) + df.columns.names[len(self.reg.columns):]
        df.rename_axis(columns=names, inplace=True)

        return df

    def iterby(self, by, pbar=None):
        groups = self.reg.groupby(by).groups.items()

        for k, uids in _optional_pbar(groups, total=len(groups), desc=by, pbar=pbar):
            subset = self.sel_mask(uids)
            subset.reg.drop(by, axis=1, inplace=True)
            yield k, subset

    @classmethod
    def concat_dict(cls, stackset_dict, names):

        merged_reg = {}

        merged_data = {}

        global_id = 0

        for key, s in stackset_dict.items():

            merged_reg[key] = s.reg.rename_axis(index='local_id')

            for local_id, data in s.data.items():
                merged_data[global_id] = data
                global_id += 1

        merged_reg = pd.concat(merged_reg, names=names)

        merged_reg = merged_reg.reset_index().drop(columns='local_id')

        return cls(merged_reg, merged_data)

    @classmethod
    def combine_idcs(cls, left, right, how='inner', left_idx_name=None, right_idx_name=None):

        left_idx_name = left_idx_name or left.reg.index.name or 'left_idx'
        left_reg: pd.DataFrame = left.reg.rename_axis(index=left_idx_name).reset_index()

        right_idx_name = right_idx_name or right.reg.index.name or 'right_idx'
        right_reg: pd.DataFrame = right.reg.rename_axis(index=right_idx_name).reset_index()

        # noinspection PyTypeChecker
        merged: pd.DataFrame = pd.merge(left_reg, right_reg, how=how)

        indices = merged[[left_idx_name, right_idx_name]].values

        results = dict(zip(merged.index.values, indices))

        return cls(merged, results)

    @classmethod
    def combine(cls, left, right, func, how='inner', left_idx_name='left_idx', right_idx_name='right_idx'):

        result = cls.combine_idcs(
            left=left, right=right,
            how=how,
            left_idx_name=left_idx_name, right_idx_name=right_idx_name,
        )

        return result.apply(
            lambda pair: func(
                left.get(pair[0]),
                right.get(pair[1]),
            )
        )
