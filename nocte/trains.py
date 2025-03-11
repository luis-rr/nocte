"""
Container for spike trains of multiple cells.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nocte import timeslice


class SpikeTrains:
    """Container for spike trains of multiple cells."""

    def __init__(self, spikes: pd.DataFrame, cells: pd.DataFrame, win_ms: timeslice.Win):
        self.spikes = spikes.copy()
        assert self.spikes.index.is_unique

        self.cells = cells.copy()
        assert self.cells.index.is_unique

        assert np.all(self.spikes['gid'].isin(self.cells.index))

        self.win_ms: timeslice.Win = timeslice.Win(*win_ms)

        if len(self.spikes) > 0:
            tmin = self.spikes['time'].min()
            tmax = self.spikes['time'].max()
            assert self.win_ms.start <= tmin, f'Spikes start too early ({timeslice.ms_to_str(tmin)}) for window {self.win_ms}'
            assert tmax <= self.win_ms.stop, f'Spikes stop too late ({timeslice.ms_to_str(tmax)}) for window {self.win_ms}'

        self._add_cells_props(time_col='time')

    @classmethod
    def from_simple_spike_table(cls, times: np.ndarray, clusters: np.ndarray, win_ms: timeslice.Win = None):
        if win_ms is None:
            win_ms = timeslice.Win(times.min(), times.max())

        return cls(
            spikes=pd.DataFrame.from_dict({
                'time': times,
                'gid': clusters,
            }),
            cells=pd.DataFrame(index=np.sort(np.unique(clusters))),
            win_ms=win_ms,
        )

    def _add_cells_props(self, time_col='time'):
        """calculate some common mentrics, per cell, from spikes"""

        self.cells['spike_count'] = self.spikes.groupby('gid').size()
        self.cells['spike_count'] = self.cells['spike_count'].fillna(0).astype(int)
        self.cells['time_quantile_95'] = self.spikes.groupby('gid')[time_col].quantile(.95)
        self.cells['time_quantile_05'] = self.spikes.groupby('gid')[time_col].quantile(.05)
        self.cells['active_duration'] = self.cells['time_quantile_95'] - self.cells['time_quantile_05']

        # this contains the site that most often is marked as "best_site" on this cell's spikes
        # however, due to drift and other sorting artifacts, it is possible that at different
        # times of the recording the best site is a different one.
        if 'best_site' in self.spikes.columns:
            self.cells['best_site'] = self.spikes.groupby(['gid', 'best_site']).size(). \
                groupby('gid').idxmax().map(lambda x: x[1])

    def describe_str(self):
        """return summary of contents"""
        return (
            f'{len(self.spikes):,d} spikes from {len(self.cells)} cells, between {self.win_ms}'
        )

    def describe(self):
        """print summary of contents"""
        print(self.describe_str())

    def _repr_html_(self):
        """pretty print on notebooks"""
        # noinspection PyProtectedMember
        return self.describe_str()

    # noinspection PyTypeChecker
    @classmethod
    def load(cls, fullpath, win_ms):
        """load HDF5"""
        return cls(
            cells=pd.read_hdf(fullpath, key='cells'),
            spikes=pd.read_hdf(fullpath, key='spikes'),
            win_ms=win_ms,
        )

    def store(self, fullpath):
        """save as HDF5"""
        self.cells.to_hdf(fullpath, key='cells')
        self.spikes.to_hdf(fullpath, key='spikes')

    def copy(self):
        """make a copy"""
        return self.__class__(
            self.spikes.copy(),
            self.cells.copy(),
            win_ms=self.win_ms,
        )

    @classmethod
    def load_spikes_jrclust(cls, fullpath, win_ms, adjust_time=None):
        """
        load the results of JRCLUST/IRONCLUST spike sorting from a csv file

        one row per spike
        columns: spike time,  cluster ID, center site ID

        :param fullpath: path to csv path
        :param win_ms:
        :param adjust_time: factor to multiply the time of the spikes
        sometimes the sorting software assumes a sampling rate of the raw data (say 30_000.3) but
        the actual sampling rate is different (say 30_000.29). This error can accumulate during a very long
        recording.
        This can be fixed by adjusting the spike times by: sorting_sr / actual_sr
        :return: 2 DataFrames (spks.cells and spikes)
        """

        spikes_raw = pd.read_csv(
            fullpath,
            names=['time', 'gid', 'best_site'])

        spikes_raw['time'] = spikes_raw['time'] * 1000.  # in ms
        spikes_raw.index.name = 'spike_id'

        cells_raw = pd.DataFrame(index=spikes_raw['gid'].unique())

        # raw data includes deleted and noisy spks.cells, drop these
        cells_raw['was_deleted'] = cells_raw.index < 0
        cells_raw['was_noise'] = cells_raw.index == 0
        cells_raw['is_good'] = ~(cells_raw['was_deleted'] | cells_raw['was_noise'])
        cells_raw.index.name = 'gid'

        spikes = spikes_raw[cells_raw['is_good'].reindex(spikes_raw['gid'], fill_value=False).values].copy()
        cells = cells_raw[cells_raw['is_good']].copy()

        if adjust_time is not None and not np.isclose(adjust_time, 1., rtol=1e-12):
            logging.warning(
                f'Adjusting spike times by {adjust_time} '
                f'which shifts last spike by {spikes.time.max() - (spikes.time.max() * adjust_time)} ms'
            )
            spikes['time'] = spikes['time'] * adjust_time

        return cls(spikes, cells, win_ms)

    @classmethod
    def load_ks2_single(cls, path, sampling_period_ms, win_ms):
        """
        load the results of Kilosort 2 spike sorting from numpy files
        """

        path = Path(path)
        time_idcs = np.load(str(path / 'spike_times.npy'))
        times = time_idcs * sampling_period_ms

        spikes = pd.DataFrame({
            'time': times,
            'gid': np.load(str(path / 'spike_clusters.npy')),
            'amplitude': np.load(str(path / 'amplitudes.npy')).squeeze(),
        })

        cells = pd.DataFrame(index=spikes['gid'].unique())

        cluster_info_path = path / 'cluster_info.tsv'
        if cluster_info_path.exists():
            cluster_info = pd.read_csv(cluster_info_path, sep='\t').set_index('id')
            assert cluster_info.index.isin(cells.index).all()
            cells = pd.concat([cells, cluster_info], axis=1).rename_axis(index='gid')

        return cls(spikes, cells, win_ms)

    @classmethod
    def concat(cls, all_spikes):
        """merge together multiple spike trains"""

        all_start = [spks.win_ms.start for spks in all_spikes]
        all_stop = [spks.win_ms.stop for spks in all_spikes]

        if not np.allclose(all_start, all_start[0]) or not np.allclose(all_stop, all_stop[0]):
            logging.warning(f'Spike trains cover different time windows')

        win_ms = timeslice.Win(np.min(all_start), np.max(all_stop))

        all_cells = pd.concat({
            i: spks.cells
            for i, spks in enumerate(all_spikes)
        }, axis=0)

        all_cells = all_cells.rename_axis(index=['_group', 'local_gid']).reset_index().rename_axis(index='gid')

        map_back = all_cells.reset_index()[['_group', 'local_gid', 'gid']].set_index(['_group', 'local_gid'])

        all_cells.drop(columns=['_group'], inplace=True)

        assert map_back.index.is_unique

        merged_spikes = []

        for i, spks in enumerate(all_spikes):
            new_spks = spks.spikes.copy()
            new_spks['gid'] = map_back.loc[i].reindex(spks.spikes['gid']).values

            merged_spikes.append(new_spks)

        merged_spikes = pd.concat(merged_spikes, axis=0, ignore_index=True)

        return cls(merged_spikes, all_cells, win_ms=win_ms)

    @classmethod
    def load_ks2_catgt_multiprobe(cls, base_path, loader):
        """
        Load the results of Kilosort 2 spike sorting from numpy files.
        As extracted from multiple probes (which got sorted separately)
        """

        all_spikes = []

        for ks_path in base_path.glob(f'catgt_*/*_imec*/imec*_ks2'):
            probe = int(str(ks_path)[-(1 + len('_ks2')):-len('_ks2')])

            spikes = cls.load_ks2_single(
                ks_path,
                loader.sampling_period,
                loader.win_ms,
            )

            spikes.cells['probe'] = probe

            all_spikes.append(spikes)

        return cls.concat(all_spikes)

    def sel(self, **props):
        """select spikes matching the given properties (for example a given gid)"""
        mask = np.ones(len(self.spikes), dtype=np.bool_)

        for k, v in props.items():
            mask &= self.spikes[k] == v

        return self.sel_mask(mask)

    def sel_isin(self, **props):
        """select spikes matching the given properties (for example a given gid)"""
        mask = np.ones(len(self.spikes), dtype=np.bool_)

        for k, v in props.items():
            mask &= self.spikes[k].isin(v)

        return self.sel_mask(mask)

    def sel_mask(self, mask):
        sel_spikes = self.spikes[mask]

        remaining_cells: pd.Index = sel_spikes['gid'].unique()
        remaining_cells = self.cells.index.intersection(remaining_cells)  # preseve the order
        sel_cells = self.cells.loc[remaining_cells]

        return self.__class__(sel_spikes, sel_cells, self.win_ms)

    def sel_between(self, **kwargs):
        """
        short-hand to select ranges along multiple dimensions

        Note time is a special property that will override the time window of the train.

        example:
            tr.select_between(time=(-5, +10))
        """
        mask = np.array([
            self.spikes[key].between(*vrange).values
            for key, vrange in kwargs.items()
        ])

        mask = np.all(mask, axis=0)

        copy = self.sel_mask(mask)

        if 'time' in kwargs:
            copy = self.__class__(
                copy.spikes,
                copy.cells,
                kwargs['time'],
            )

        return copy

    def drop(self, **props):
        """drop spikes matching the given properties (for example a given gid)"""
        mask_spikes = np.ones(len(self.spikes), dtype=np.bool_)

        for k, v in props.items():
            mask_spikes &= self.spikes[k] == v

        return self.sel_mask(~mask_spikes)

    def sel_cells(self, **props):
        """select cells (and their corresponding spikes) matching the given properties (for example a brain region)"""
        mask_cells = np.ones(len(self.cells), dtype=np.bool_)

        for k, v in props.items():
            mask_cells &= self.cells[k] == v

        return self.sel_cells_mask(mask_cells)

    def sel_cells_mask(self, mask):
        """select cells (and their corresponding spikes) using a boolean mask"""
        sel_cells = self.cells[mask]

        mask_spikes = self.spikes['gid'].isin(sel_cells.index)
        sel_spikes = self.spikes[mask_spikes]

        return self.__class__(sel_spikes, sel_cells, self.win_ms)

    def sel_cells_isin(self, **props):
        """select cells (and their corresponding spikes) matching the given properties (for example a brain region)"""
        mask_cells = np.ones(len(self.cells), dtype=np.bool_)

        for k, v in props.items():
            mask_cells &= self.cells[k].isin(v)

        return self.sel_cells_mask(mask_cells)

    def reset_gids(self):
        """
        Reassing consecutive gids to the cells.
        Useful after sorting or selecting.
        """

        mapback = pd.Series(np.arange(len(self.cells)), self.cells.index)

        cells = self.cells.set_index(mapback.values)
        spikes = self.spikes.copy()
        spikes['gid'] = self.spikes['gid'].map(mapback).values

        return self.__class__(
            spikes,
            cells,
            self.win_ms,
        )

    def sort_reset_gids(self, by=('probe', 'depth', 'ch')):
        """
        Sort cells by some properties and reset the gids.
        Useful for plotting cells sorted by depth.
        """
        by = list(by)

        copy = self.__class__(
            self.spikes,
            self.cells.sort_values(by),
            self.win_ms,
        )

        return copy.reset_gids()

    def iter_cells_by(self, by):
        """
        Iterate by groups of cells defined by some property.
        """
        for (key, scells) in self.cells.groupby(by):
            yield key, self.sel_in(gid=scells.index)

    def shift_time(self, ms):
        """Add a time offset to all spikes. Returns a copy"""
        spikes_copy = self.spikes.copy()
        spikes_copy['time'] += ms

        return self.__class__(
            spikes_copy,
            self.cells,
            win_ms=self.win_ms.shift(ms),
        )

    def unadjust_times_by_probe_sampling(self, raw_loader, ref_probe=0):
        """
        In neuropixel recordings, different probes have slightly different sampling rates.
        CatGT (from the spike sorting pipeline) will adjust all spike times to be relative
        to one of the probles (often probe=0).
        In order to load the raw LFP (non-interpolated), we need to undo this adjustment.

        Note that this allows you to load LFP and process it, but you will need to
        re-apply the adjustment to the result if you want to do comparisons across probes.

        :return: a series with the un-adjusted spike times
        """

        probe_periods = {
            probe: probe_loader.sampling_period
            for probe, probe_loader in raw_loader.loaders.items()
        }

        factors = self.cells['probe'].map(probe_periods) / raw_loader.loaders[ref_probe].sampling_period

        factors = self.spikes['gid'].map(factors)

        times = self.spikes['time'] * factors

        return times

    def compute_activity_per_cluster(
            self, tbins, gid_col='gid', time_col='time',
            fr=False,
            rolling_wins=None,
            rolling_win_type='hamming',
            pbar=True
    ):
        """return a pd.DataFrame containing the histogram of spikes for each cluster"""

        grouped = self.spikes.groupby(gid_col)[time_col]
        if pbar:
            grouped = tqdm(grouped, desc='spks.cells')

        df = pd.DataFrame.from_dict({
            gid: np.histogram(times, bins=tbins, density=False)[0]
            for gid, times in grouped},
        )

        # noinspection PyUnresolvedReferences
        tbins_index = pd.IntervalIndex.from_breaks(tbins)
        df = df.set_index(tbins_index).rename_axis(index=time_col, columns=gid_col)

        if fr:
            bin_widths = df.index.length * timeslice.MS_TO_S
            df = (df.T / bin_widths).T

        if rolling_wins is not None:
            df = df.rolling(window=rolling_wins, center=True, win_type=rolling_win_type).mean().dropna()

        return df

    def compute_isi(self, tbins, time_col='time', pmf=False, pbar=True):
        """Compute Inter-Spike-Intervals"""
        tdiffs = self.spikes.sort_values(['gid', time_col]).groupby(['gid'])[time_col].diff()
        bin_widths = tbins[1:] - tbins[:-1]

        grouped = tdiffs.groupby(self.spikes['gid'])
        if pbar:
            grouped = tqdm(grouped, desc='spks.cells')

        # noinspection PyUnresolvedReferences
        tbins_index = pd.IntervalIndex.from_breaks(tbins)

        isis = pd.DataFrame.from_dict({
            gid: np.histogram(td, bins=tbins)[0] / bin_widths
            for gid, td in grouped
        }).set_index(tbins_index).rename_axis(index='tbin', columns='gid')

        if pmf:
            isis = isis / isis.sum()

        return isis

    def get_counts(self) -> pd.Series:
        """Get number of spikes per cell"""
        return self.spikes['gid'].value_counts().reindex(self.cells.index, fill_value=0)

    def to_mua(self, by='probe'):
        """
        Produce multi-unit activity.
        Reassign all spikes that belong to groups of cells to a new common group gid.
        Cells are grouped by column 'by'. This collapses the cells table.
        """
        spikes = self.spikes.copy()

        for key, gids in self.cells.groupby(by).groups.items():
            spikes.loc[self.spikes['gid'].isin(gids), 'gid'] = key

        return self.__class__(
            spikes=spikes,
            cells=pd.DataFrame([], index=self.cells[by].unique()),
            win_ms=self.win_ms,
        )
