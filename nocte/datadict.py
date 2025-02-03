import h5py
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

from nocte.df_wrapper import DataFrameWrapper
from nocte.stacks import Stack


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

    def to_hdf(self, filename, desc='data'):
        """
        Store data and registry to HDF5.
        Data must implement "to_hdf" (e.g. pd.DataFrame)
        """
        self.reg.to_hdf(filename, key='reg')

        for k, v in self.data.items():
            if hasattr(v, 'to_hdf'):
                key = f'{desc}_{k:06d}'
                v.to_hdf(filename, key=key)

            # elif hasattr(v, 'store_hdf'):
            #     key = f'stack_{k:06d}'
            #     v.store_hdf(filename, key=key)

            else:
                raise NotImplementedError()

    @classmethod
    def from_hdf(cls, filename, desc='data', show_pbar=True):
        """
        Load data and registry from HDF5.
        """
        # noinspection PyTypeChecker
        reg: pd.DataFrame = pd.read_hdf(filename, key='reg')

        with h5py.File(str(filename), mode='r') as f:
            file_keys = list(f.keys())

        data = {}
        index = reg.index
        if show_pbar:
            index = tqdm(reg.index, desc='loading')

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

    def get(self, **kwargs):
        """Select a SINGLE stack and return it"""
        if len(kwargs) == 0:
            selected = self
        else:
            selected = self.sel(**kwargs)

        assert len(selected) > 0, f'No items selected'
        assert len(selected) == 1, f'Multiple items selected ({len(selected):,g})'
        return list(selected.data.values())[0]

    def items(self, pbar=True):
        index = self.reg.index

        pbar_desc = ''
        if isinstance(pbar, str):
            pbar_desc = pbar
            pbar = True

        if pbar:
            index = tqdm(index, total=len(self.reg), desc=pbar_desc)

        for uid in index:
            yield self.reg.loc[uid], self.data[uid]

    def extract_df(self, func, pbar=True, **kwargs) -> pd.DataFrame:
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

    def iterby(self, by, show_pbar=True):
        groups = self.reg.groupby(by).groups.items()

        if show_pbar:
            groups = tqdm(groups, total=len(groups), desc=by)

        for k, uids in groups:
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
