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
from nocte.traces import Traces
from nocte.datadict import DataDict
from nocte.timeslice import Win

logger = logging.getLogger(__name__)


def _get_sampling_rate(ks_path):
    # from SpikeGLX metadata
    meta_files = list(ks_path.glob("*.ap.meta"))
    assert len(meta_files) == 1

    meta_path = meta_files[0]

    meta = {}
    with open(meta_path, "r") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v

    return float(meta["imSampRate"])


def _load_kilosort4_folder(ks_path: str | Path):
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

    spike_times = np.load(ks_path / "spike_times.npy").squeeze()
    spike_clusters = np.load(ks_path / "spike_clusters.npy").squeeze().astype(int)

    spikes = pd.DataFrame({
        "ref_time": spike_times,
        "unit_id": spike_clusters,
    })

    # Optional, aligned spike-level fields
    for name in ["amplitudes", "spike_templates"]:
        f = ks_path / f"{name}.npy"
        if f.exists():
            spikes[name] = np.load(f).squeeze()

    # ----------------------------------------------------
    # Units

    # KS tables
    ks_table = []
    for fname in [
        "cluster_group.tsv",
        "cluster_KSLabel.tsv",
        "cluster_Amplitude.tsv",
        "cluster_ContamPct.tsv",
    ]:
        f = ks_path / fname
        if f.exists():
            df = pd.read_csv(f, sep="\t").set_index("cluster_id")
            ks_table.append(df)

    ks_table = pd.concat(ks_table, axis=1)
    ks_table.index = ks_table.index.astype(int)

    # Phy manual curation:
    info_file = ks_path / "cluster_info.tsv"
    phy_table = pd.read_csv(info_file, sep="\t").set_index("cluster_id")
    phy_table.index = phy_table.index.astype(int)

    # combine them safely
    all_ids = np.sort(ks_table.index.union(phy_table.index))
    ks_table = ks_table.reindex(all_ids)
    phy_table = phy_table.reindex(all_ids)

    units: pd.DataFrame = phy_table.copy()

    for col, vals in ks_table.items():

        if col not in units.columns:
            units[col] = vals

        else:
            if not units[col].equals(vals):
                logging.warning(f'Different contents for Phy and KS column: "{col}"')
                units[f'{col}_ks'] = vals

    # ----------------------------------------------------
    # Sanity checks:

    missing = set(spikes.unit_id.unique()) - set(units.index)
    if missing:
        raise RuntimeError(f"Spikes reference missing unit_ids: {missing}")

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

    def rate_rolling_gauss(
        self,
        *,
        sigma: float,
        by: str = "ref_time",
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

        spikes_split = DataDict.from_split(self._spikes, by="unit_id")

        dd_rates = spikes_split.apply(
            lambda sp:
                sp.rate_rolling_gauss(
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
        missing = set(reg['unit_id'].unique()) - set(units.index)
        if missing:
            raise ValueError(f'Spikes reference unknown unit_id(s): {missing}')

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
        valid_win = Win(
            0,
            sample_count / sampling_rate * 1000
        )

        return cls(
            reg=spikes_df,
            units=units_df,
            win_ms=valid_win,
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
