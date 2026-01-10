"""
Container for spike trains of multiple cells.
"""
from tqdm.auto import tqdm
from nocte import timeslice


import logging
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
from nocte.df_wrapper import DataFrameWrapper
from nocte.events import Events

logger = logging.getLogger(__name__)


def _load_kilosort_folder(folder: Path):
    folder = Path(folder)

    # -------------------------------------------
    # spikes
    spike_sample_idcs = np.load(folder / 'spike_times.npy').squeeze()
    spike_clusters = np.load(folder / 'spike_clusters.npy').squeeze()

    if spike_sample_idcs.ndim != 1 or spike_clusters.ndim != 1:
        raise ValueError('Spike arrays must be 1D')

    if len(spike_sample_idcs) != len(spike_clusters):
        raise ValueError('spike_times and spike_clusters length mismatch')

    spikes_df = pd.DataFrame({
        'ref_idx': spike_sample_idcs,
        'unit_id': spike_clusters,
    })

    # -------------------------------------------
    # units
    if (folder / 'cluster_info.tsv').exists():
        units_df = pd.read_csv(folder / 'cluster_info.tsv', sep='\t')

    elif (folder / 'cluster_group.tsv').exists():
        units_df = pd.read_csv(folder / 'cluster_group.tsv', sep='\t')

    else:
        raise FileNotFoundError('No cluster_info.tsv or cluster_group.tsv found')

    if 'cluster_id' not in units_df.columns:
        raise ValueError('Cluster table must contain cluster_id column')

    units_df = units_df.set_index('cluster_id')
    units_df.index.name = 'unit_id'

    return spikes_df, units_df



class _UnitsView(DataFrameWrapper):
    """
    Relational view over units inside a Spikes object.

    Selection methods operate on the units table, but return
    a new Spikes object with both units and spikes filtered.
    """

    def __init__(self, spikes: 'Spikes'):
        self._spikes = spikes
        super().__init__(spikes.units.reg)

    def _apply_mask(self, mask) -> 'Spikes':
        """
        Apply a unit-level mask and return a new Spikes
        with both units and spikes filtered accordingly.
        """
        new_units = self.reg.loc[mask]

        spikes_mask = self._spikes.reg['unit_id'].isin(new_units.index)
        new_spikes = self._spikes.sel_mask(spikes_mask)

        return self._spikes.__class__(
            reg=new_spikes.reg,
            units=new_units,
            win_ms=new_spikes.win_ms,
        )

    def get_count(self) -> pd.Series[int]:
        counts = self._spikes.reg.groupby('unit_id').size()
        counts = counts.reindex(self.index, fill_value=0)
        return counts

    def drop_silent(self):
        """
        Drop units with zero spikes.
        """
        counts = self.get_count()
        return self.sel_mask(counts > 0)

    def set_index(self, idx):
        old_idcs = self.reg.index
        new_units = self.reg.set_index(idx)

        mapping = pd.Series(new_units.index, index=old_idcs)
        new_spikes = self._spikes.reg.copy()
        new_spikes['unit_id'] = new_spikes['unit_id'].map(mapping)

        if new_spikes['unit_id'].isna().any():
            raise ValueError("Unit ID remapping produced NaNs")

        return self._spikes.__class__(
            reg=new_spikes,
            units=new_units,
            win_ms=self._spikes.win_ms,
        )

    def reset_index(self):
        return self.set_index(
            np.arange(len(self.reg.index))
        )


class Spikes(Events):
    """
    Collection of spikes and associated units.
    """
    def __init__(self, reg: pd.DataFrame, units: pd.DataFrame, win_ms):
        missing = set(reg['unit_id'].unique()) - set(units.index)
        if missing:
            raise ValueError(f'Spikes reference unknown unit_id(s): {missing}')

        self.units = DataFrameWrapper(units)
        self.win_ms = win_ms
        super().__init__(reg)

    @classmethod
    def load_kilosort(
            cls,
            folder,
            win_ms,
            *,
            sampling_rate,
    ):
        spikes_df, units_df = _load_kilosort_folder(folder)

        spikes_df['ref_time'] = spikes_df['ref_idx'] / sampling_rate * 1000.0
        spikes_df.drop('ref_idx', inplace=True)

        return cls(
            reg=spikes_df,
            units=units_df,
            win_ms=win_ms,
        )

    @classmethod
    def load_hdf(cls, path, key='sp'):
        return cls(
            reg=pd.read_hdf(path, key=f'{key}_spikes'),
            units=pd.read_hdf(path, key=f'{key}_units'),
            win_ms=tuple(pd.read_hdf(path, key=f'{key}_win_ms')),
        )

    def store_hdf(self, path, key='sp'):
        self.reg.to_hdf(path, key=f'{key}_spikes')
        self.units.reg.to_hdf(path, key=f'{key}_units')
        pd.Series(self.win_ms).to_hdf(path, key=f'{key}_win_ms')

    @property
    def by_unit(self):
        return _UnitsView(self)

    def _apply_mask(self, mask) -> Self:
        """
        Apply a spike-level mask and return a new Spikes
        with spikes filtered accordingly.
        Note this leves behind units with potentially
        zero spikes by design. Use by_unit.drop_silent() to remove those.
        """
        sel_spikes = self.reg.loc[mask]

        return self.__class__(
            reg=sel_spikes,
            units=self.units.reg,
            win_ms=self.win_ms,
        )
