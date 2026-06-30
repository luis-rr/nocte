"""
Container for spike trains of multiple cells.
"""

import logging
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

from nocte import timeslice
from nocte.datadict import DataDict
from nocte.df_wrapper import DataFrameWrapper
from nocte.events import Events
from nocte.timeslice import Win
from nocte.traces import Traces

logger = logging.getLogger(__name__)


def _get_sampling_rate(ks_path):
    # from SpikeGLX metadata
    meta_files = list(ks_path.glob('*.ap.meta'))
    assert len(meta_files) == 1

    meta_path = meta_files[0]

    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                meta[k] = v

    return float(meta['imSampRate'])


def _load_kilosort4_folder(ks_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    spikes : pd.DataFrame
        One row per spike with columns:
            - ref_time in ms
            - unit_id
    units : pd.DataFrame
        Indexed by unit_id with cluster metadata and labels.
    """

    ks_path = Path(ks_path)

    # ----------------------------------------------------
    # Spike times:

    spike_times = np.load(ks_path / 'spike_times.npy').squeeze()
    spike_clusters = np.load(ks_path / 'spike_clusters.npy').squeeze().astype(int)

    spikes = pd.DataFrame(
        {
            'ref_time': spike_times,
            'unit_id': spike_clusters,
        }
    )

    # Optional, aligned spike-level fields
    for name in ['amplitudes', 'spike_templates']:
        f = ks_path / f'{name}.npy'
        if f.exists():
            spikes[name] = np.load(f).squeeze()

    # ----------------------------------------------------
    # Units

    cluster_paths = list(ks_path.glob('cluster_*.tsv'))
    if not cluster_paths:
        raise RuntimeError(f'No cluster_*.tsv files found in {ks_path}')

    cluster_table = pd.concat(
        {
            path.name[len('cluster_') : -len('.tsv')]: pd.read_csv(
                path,
                sep='\t',
            ).set_index('cluster_id')
            for path in cluster_paths
        },
        axis=1,
    )

    units_cols = {}
    for col in cluster_table.columns.get_level_values(1).unique():
        entries: pd.DataFrame = cluster_table.swaplevel(axis=1).loc[:, col]  # type: ignore

        entries = entries.T.drop_duplicates().T

        while len(entries.columns) > 1:
            a = entries.iloc[:, 0]
            b = entries.iloc[:, 1]

            both_present = a.notna() & b.notna()

            same_where_present = a[both_present].equals(b[both_present])

            if not same_where_present:
                logger.warning(
                    f'Column "{col}" in "{entries.columns[0]}" and "{entries.columns[1]}" disagree where both are present, keeping first.'
                )

            combined = a.combine_first(b)

            entries = pd.concat(
                [combined]
                + [entries.iloc[:, i] for i in range(2, len(entries.columns))],
                axis=1,
            )

        units_cols[col] = entries.iloc[:, 0]

    units: pd.DataFrame = pd.concat(units_cols, axis=1)

    # ----------------------------------------------------
    # Sanity checks:

    missing = set(spikes.unit_id.unique()) - set(units.index)
    if missing:
        raise RuntimeError(f'Spikes reference missing unit_ids: {missing}')

    return spikes, units


class _UnitsView(DataFrameWrapper):
    """
    Relational view over units inside a Spikes object.

    Selection methods operate on the units table, but return
    a new Spikes object with both units and spikes filtered.
    """

    def __init__(self, reg, spikes: 'Spikes'):
        self._spikes = spikes
        super().__init__(reg)

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

    def get_counts(self) -> pd.Series:
        counts = self._spikes.reg.groupby('unit_id').size()
        counts = counts.reindex(self.index, fill_value=0)
        return counts

    def get_inter_spike_intervals(self) -> dict[int, pd.Series]:
        """
        Returns
        -------
        dict[int, pd.Series]
            unit_id -> pd.Series of inter-spike intervals in ms
        """
        return {
            unit_id: times.sort_values().diff().dropna()
            for unit_id, times in self._spikes.reg.groupby('unit_id')['ref_time']
        }  # type: ignore

    def get_inter_spike_intervals_dists(self, bins: np.ndarray | None = None) -> Traces:

        isis = self._spikes.by_units.get_inter_spike_intervals()

        if bins is None:
            tmax = max(isi.max() for isi in isis.values())
            tbins = np.arange(0, tmax)
        else:
            tbins = np.asarray(bins)

        bin_centers = (tbins[:-1] + tbins[1:]) * 0.5

        isi_dists = {
            unit_id: np.histogram(isi, bins=tbins)[0] for unit_id, isi in isis.items()
        }

        isi_dists = pd.DataFrame(
            isi_dists,
            index=bin_centers,
        )

        return Traces(reg=self._spikes.units, traces=isi_dists)

    def get_counts_in_bins(self, bins: np.ndarray):

        times = self._spikes.reg['ref_time'].to_numpy()
        units_all = self._spikes.reg['unit_id'].to_numpy()

        # map spikes to bin indices
        bin_idcs = np.digitize(times, bins) - 1
        n_bins = len(bins) - 1

        # validity mask
        valid = (bin_idcs >= 0) & (bin_idcs < n_bins)

        # ---- warning block ----
        n_total = len(times)
        n_invalid = np.count_nonzero(~valid)

        if n_invalid > 0:
            logging.warning(
                f'{n_invalid:,d}/{n_total:,d} spikes fell outside bin range [{bins[0]:.6g}, {bins[-1]:.6g})'
            )

        # keep valid spikes
        bin_idcs = bin_idcs[valid]
        unit_ids = units_all[valid]

        # mapping
        units = np.sort(self._spikes.units.index.to_numpy())

        unit_lookup = np.zeros(units.max() + 1, dtype=np.int32)
        unit_lookup[units] = np.arange(len(units))
        unit_idx = unit_lookup[unit_ids]

        # accumulate
        binned = np.zeros((n_bins, len(units)), dtype=np.int16)
        np.add.at(binned, (bin_idcs, unit_idx), 1)

        # centers
        bin_centers = (bins[:-1] + bins[1:]) * 0.5

        binned_df = pd.DataFrame(
            binned,
            index=pd.Index(bin_centers, name='ref_time'),
            columns=pd.Index(units, name='unit_id'),
        )

        return Traces(
            reg=self._spikes.units.rename_axis(
                index='unit_id',
            ),
            traces=binned_df,
        )

    def drop_silent(self):
        """
        Drop units with zero spikes.
        """
        counts = self.get_counts()
        return self.sel_mask(counts > 0)

    def set_index(self, idx):
        old_idcs = self.reg.index
        new_units = self.reg.set_index(idx)

        mapping = pd.Series(new_units.index, index=old_idcs)
        new_spikes = self._spikes.reg.copy()
        new_spikes['unit_id'] = new_spikes['unit_id'].map(mapping)

        if new_spikes['unit_id'].isna().any():
            raise ValueError('Unit ID remapping produced NaNs')

        return self._spikes.__class__(
            reg=new_spikes,
            units=new_units,
            win_ms=self._spikes.win_ms,
        )

    def reset_index(self):
        return self.set_index(np.arange(len(self.reg.index)))

    def rate_rolling_gauss(
        self,
        *,
        sigma: float,
        by: str = 'ref_time',
        step: int = 1_000,
        win_ms=None,
        pbar=None,
    ):
        """
        Estimate instantaneous firing rate per unit using a Gaussian kernel.

        Returns
        -------
        Traces
            One trace per unit_id, indexed by time.
        """
        win_ms = win_ms or self._spikes.win_ms

        spikes_split = DataDict.from_split(self._spikes, by='unit_id')

        dd_rates = spikes_split.apply(
            lambda sp: sp.rate_rolling_gauss(
                sigma=sigma,
                valid_win=win_ms,
                by=by,
                step=step,
            ),
            pbar=pbar,
        )

        dd_rates = dd_rates.set_index('unit_id')

        traces_reg = dd_rates.reg.join(self.reg)

        return Traces(
            traces_reg,
            pd.DataFrame(dd_rates.data),
        )


class Spikes(Events):
    """
    Collection of spikes and associated units.
    """

    def __init__(self, reg: pd.DataFrame, units: pd.DataFrame, win_ms: Win | tuple):

        win_ms = Win(*win_ms)

        if win_ms.length < 0:
            raise ValueError(f'Negative extaction window: {win_ms}')

        missing = set(reg['unit_id'].unique()) - set(units.index)
        if missing:
            raise ValueError(f'Spikes reference unknown unit_id(s): {missing}')

        inside = reg['ref_time'].between(*win_ms)
        if not inside.all():
            actual_win = Win(reg['ref_time'].min(), reg['ref_time'].max())
            logger.error(
                f'Events outside extraction window. Found: {actual_win} Expected: {win_ms}'
            )

        self.units = units
        self.win_ms = Win(*win_ms)
        super().__init__(reg)

    def _replace_reg(self, reg) -> Self:
        """
        Make a copy and pass over metadata.
        """
        return self.__class__(
            reg,
            units=self.units,
            win_ms=self.win_ms,
        )

    @classmethod
    def load_kilosort(
        cls,
        folder,
        *,
        sampling_rate,
        sample_count,
    ):
        spikes_df, units_df = _load_kilosort4_folder(folder)
        spikes_df: pd.DataFrame
        units_df: pd.DataFrame
        win_ms: tuple

        # drop statistics that should be recalculated on the fly
        units_df.drop(['fr', 'n_spikes'], axis=1, errors='ignore', inplace=True)
        units_df['KSLabel'] = units_df['KSLabel'].fillna('unknown').astype('string')
        units_df['group'] = units_df['group'].fillna('unknown').astype('string')

        # convert times from sample idcs to ms
        spikes_df['ref_time'] = spikes_df['ref_time'] / sampling_rate * 1000
        valid_win = Win(0, sample_count / sampling_rate * 1000)

        return cls(
            reg=spikes_df,
            units=units_df,
            win_ms=valid_win,
        )

    @classmethod
    def from_hdf(cls, path, key='sp'):  # TODO Deprecate in base class
        return cls.load_hdf(path, key=key)

    def to_hdf(self, path, key='sp'):  # TODO Deprecate in base class
        return self.store_hdf(path, key=key)

    @classmethod
    def load_hdf(cls, path, key='sp'):
        return cls(
            reg=pd.read_hdf(path, key=f'{key}_spikes'),
            units=pd.read_hdf(path, key=f'{key}_units'),
            win_ms=tuple(pd.read_hdf(path, key=f'{key}_win_ms')),
        )

    def store_hdf(self, path, key='sp'):
        self.reg.to_hdf(path, key=f'{key}_spikes')
        self.units.to_hdf(path, key=f'{key}_units')
        pd.Series(self.win_ms).to_hdf(path, key=f'{key}_win_ms')

    @property
    def by_units(self):
        return _UnitsView(self.units, self)

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
            units=self.units,
            win_ms=self.win_ms,
        )

    def describe(self, quiet=False) -> str:
        """
        Short, human-readable description.
        :param quiet: if False the description is printed
        """
        desc = (
            f'{len(self.index):,g} spikes from '
            f'{len(self.units.index):,g} units in '
            f'{timeslice.ms_to_str(self.win_ms.length)}'
        )

        if not quiet:
            print(desc)

        return desc

    def crop(self, win: timeslice.Win, on='ref_time') -> Self:
        """
        Extract the events within windows after matching them by a column.
        Events in no window are dropped.
        """

        if win.start < self.win_ms.start or self.win_ms.stop < win.stop:
            logger.warning(
                f'Cropping window {win} is wider than event extraction window {self.win_ms}'
            )
            win = timeslice.Win(
                max(self.win_ms.start, win.start),
                min(self.win_ms.stop, win.stop),
            )

        cropped = self.sel_between(**{on: win})

        return self.__class__(
            reg=cropped.reg,
            units=cropped.units,
            win_ms=self.win_ms,
        )

    def count_in_wins(self, trials):
        counts_in_trial = pd.Series(
            {
                unit_id: len(trials.classify_events(s['ref_time']))
                for unit_id, s in self.iter_groupby('unit_id', pbar=False)
            }
        )
        counts_in_trial = counts_in_trial.reindex(self.units.index, fill_value=0)

        return counts_in_trial

    def copy(self):
        return self.__class__(
            self.reg.copy(),
            self.units.copy(),
            self.win_ms,
        )

    def relabel_units(self, mapping):
        spikes = self.copy()
        assert spikes.units.index.isin(mapping.index).all()
        assert mapping.index.is_unique
        assert mapping.is_unique

        spikes.units.index = spikes.units.index.map(mapping)
        assert spikes.units.index.is_unique
        spikes.units.sort_index(inplace=True)

        spikes.reg['unit_id'] = mapping.reindex(spikes.reg['unit_id']).values

        return spikes

    def relabel_units_by(self, score: pd.Series, ascending=True):
        mapping = pd.Series(
            np.arange(len(score)),
            index=score.sort_values(ascending=ascending).index,
        )

        return self.relabel_units(mapping)

    def relabel_units_by_trial_selectivity(self, trials):

        counts = pd.DataFrame(
            {
                'in_trial': self.count_in_wins(trials),
                'in_rest': self.count_in_wins(trials.invert()),
            }
        )

        counts['selectivity'] = counts['in_trial'] / counts[
            ['in_trial', 'in_rest']
        ].sum(axis=1)

        spikes_sorted = self.relabel_units_by(counts['selectivity'])

        return spikes_sorted
