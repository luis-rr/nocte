import h5py
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

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

    @classmethod
    def from_dict(cls, data: dict, names=None):
        reg = pd.DataFrame.from_records(list(data.keys()), columns=names)
        data_mapped = {uid: data[tuple(k)] for uid, *k in reg.itertuples()}
        return cls(reg, data_mapped)

    def to_hdf(self, filename, desc='data', pbar=None):
        """
        Store data and registry to HDF5.
        Data must implement "to_hdf" (e.g. pd.DataFrame)
        """
        self.reg.to_hdf(filename, key='reg')

        for k, v in self.items(pbar=pbar):
            if hasattr(v, 'to_hdf'):
                key = f'{desc}_{k:06d}'
                v.to_hdf(filename, key=key)

            # elif hasattr(v, 'store_hdf'):
            #     key = f'stack_{k:06d}'
            #     v.store_hdf(filename, key=key)

            else:
                raise NotImplementedError()

    @classmethod
    def from_hdf(cls, filename, desc='data', pbar=True):
        """
        Load data and registry from HDF5.
        """
        # noinspection PyTypeChecker
        reg: pd.DataFrame = pd.read_hdf(filename, key='reg')

        with h5py.File(str(filename), mode='r') as f:
            file_keys = list(f.keys())

        data = {}
        index = _optional_pbar(reg.index, total=len(reg.index), desc='loading', pbar=pbar)

        for k in index:

            key = f'{desc}_{k:06d}'
            if key in file_keys:
                data[k] = pd.read_hdf(filename, key=key)
                continue

            # fmt = 'stack'
            # key = f'{fmt}_{k:06d}'
            #
            # if f'{key}_values' in file_keys:
            #     data[k] = Stack.load_hdf(filename, key=key)
            #     continue

            raise KeyError(f'Missing data for {key}. Found: {file_keys}')

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

    def get(self, idx=None) -> pd.Series:
        """return a single item. If no index it's given, we assume there is only one"""
        if idx is None:
            assert len(self.index) == 1, f'Found too many traces:\n{self.reg}'
            idx = self.index[0]

        return self.data[idx]

    def items(self, pbar=None):
        """
        returns an iterator to go over each data object:

            for k, data in dd.items(pbar=True):
                pass

        """
        # Note we want to respect the order of the registry, not of the dict
        for k in _optional_pbar(self.index, total=len(self.index), pbar=pbar):
            yield k, self.data[k]

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
