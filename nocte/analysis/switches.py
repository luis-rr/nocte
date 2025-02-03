"""
Code to analyse the switches of inter-hemispheric dominance.
"""
import logging

import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from nocte import plot as splot
from nocte import stacks, timeslice, events
from nocte.timeslice import ms


def extract_lead_traces(xcorr, rem_wins, all_beta, lag_modes):
    lead_traces: pd.DataFrame = xcorr.sel(lag=list(lag_modes)).to_dataframe2d().T
    lead_traces.columns = ['lead_ch1', 'lead_ch0']

    lead_traces['lead_diff'] = lead_traces['lead_ch1'] - lead_traces['lead_ch0']

    # we extract power at a lower sampling rate, so make sure we cover the same space
    lead_traces = rem_wins.get_global_win().crop_df(lead_traces)
    lead_traces['beta_ch0'] = events.interpolate_trace(all_beta['ch0'], lead_traces.index)
    lead_traces['beta_ch1'] = events.interpolate_trace(all_beta['ch1'], lead_traces.index)
    lead_traces['rem_state'] = rem_wins.generate_cat(lead_traces.index)
    lead_traces['log_lead_ch0'] = np.log10(lead_traces['lead_ch0'])
    lead_traces['log_lead_ch1'] = np.log10(lead_traces['lead_ch1'])

    return lead_traces


def load_lead_traces(
        reg, exp_name,
        tau=1_000,
        exp_valid_win=timeslice.Win(ms(hours=2), ms(hours=2 + 9)),
        low_hz=100,
        suffix='_exp_clipped',
):
    all_beta = reg.load_all_beta_norm(exp_name, exp_valid_win=exp_valid_win)

    rem_wins = extract_rem_wins_from_log_beta_thresh_detour(all_beta)

    xcorr = load_xcorr(reg, exp_name, tau=tau, suffix=suffix, exp_valid_win=exp_valid_win, low_hz=low_hz)

    lag_modes = extract_lag_modes(xcorr)

    lead_traces = extract_lead_traces(xcorr, rem_wins, all_beta, lag_modes)
    lead_traces['lead_state_thresh'] = classify_lead_state_by_diag_thresh(lead_traces)
    lead_traces['lead_state'] = smooth_lead_state_detour(lead_traces['lead_state_thresh'])

    return lead_traces


def extract_lag_modes(xcorr, min_lag=5):
    best_lags = xcorr.idxmax('lag').to_series()

    # avoid boundary conditions
    best_lags = best_lags[best_lags != xcorr.coords['lag'].max()]
    best_lags = best_lags[best_lags != xcorr.coords['lag'].min()]

    lag_modes = (
        best_lags[best_lags <= -min_lag].mode().max(),
        best_lags[best_lags >= +min_lag].mode().min(),
    )

    if lag_modes[0] == -min_lag or lag_modes[1] == +min_lag:
        logging.warning(f'Lags too small? {lag_modes}')

    return lag_modes


def load_xcorr(reg, exp_name, tau, suffix='_exp', exp_valid_win=None, low_hz=100):
    sliding_win = tau * 10

    xcorr = stacks.Stack.load_hdf(
        reg.get_path_xcorr_area(exp_name, area='CLA', sliding_win=sliding_win, suffix=suffix, low_hz=low_hz),
        'xcorr'
    )

    if exp_valid_win is not None:
        xcorr = xcorr.sel_between(time=exp_valid_win)
        if 0 in xcorr.shape:
            logging.warning(f'x-corr does not exist in window {exp_valid_win}')
            return xcorr

    vmax = np.quantile(xcorr.values.ravel(), .999)
    assert vmax > 0
    xcorr = xcorr / vmax

    return xcorr


def extract_rem_wins_from_log_beta_thresh_detour(all_beta, beta_thresh=.15, max_detours=ms(seconds=15)):
    is_rem = all_beta['beta_max'] > beta_thresh

    rem_states = is_rem.map({True: 'rem', False: 'sws'})

    rem_wins = timeslice.Windows.build_from_contiguous_values(rem_states, include_right=False)

    rem_wins = rem_wins.merge_sandwiched(max_length=max_detours)

    return rem_wins


def classify_lead_state_by_diag_thresh(lead_traces, col0='log_lead_ch0', col1='log_lead_ch1', slope=1.2):
    lead_states = pd.Series('x', index=lead_traces.index)

    sws_center = lead_traces.loc[lead_traces['rem_state'] == 'sws', [col0, col1]].mean()

    mask = (lead_traces[col0] - sws_center[col0]) * slope + sws_center[col0] < lead_traces[col1]
    lead_states.loc[mask] = 'lead_ch1'

    mask = (lead_traces[col1] - sws_center[col1]) * slope + sws_center[col1] < lead_traces[col0]
    lead_states.loc[mask] = 'lead_ch0'

    lead_states.loc[lead_traces['rem_state'] == 'sws'] = 'sws'

    return lead_states


def extract_cycle_wins(rem_wins, rem_align=.5, sws_align=None):
    """
    Create windows that go from SWS to SWS including exactly one REM
    in each window.
    Reference time is extracted relative to this REM center.
    """

    if sws_align is None:
        sws_align = 1 - rem_align

    rem = rem_wins.sel(cat='rem')

    sws = rem_wins.sel(cat='sws')
    breaks = sws.relative_time(sws_align)

    if rem_wins['cat'].values[0] == 'rem':
        breaks = np.append(rem_wins['start'].min(), breaks)

    if rem_wins['cat'].values[-1] == 'rem':
        breaks = np.append(breaks, rem_wins['stop'].max())

    cycle_wins = timeslice.Windows.build_between(breaks)

    cycle_wins['ref'] = rem.relative_time(rem_align).values
    assert cycle_wins.is_ref_inside().all()
    cycle_wins['rem_idx'] = rem.index

    return cycle_wins.rename_index('cycle_idx')


def classify_wins_by_cycle(lead_wins_raw, cycle_wins):
    lead_wins: timeslice.Windows = lead_wins_raw.crop_to_multiple(cycle_wins, drop=True)
    lead_wins = lead_wins.sel_mask(lead_wins.lengths() > 0)
    lead_wins['ref'] = lead_wins.mid()

    lead_wins: pd.DataFrame = cycle_wins.rename_index('idx').annotate_events(lead_wins.wins, col='ref', prefix='cycle')
    lead_wins.rename(columns=dict(cycle_rem_idx='rem_idx'), inplace=True)
    lead_wins: pd.DataFrame = lead_wins.dropna()
    lead_wins['cycle_idx'] = lead_wins['cycle_idx'].astype(int)
    lead_wins = timeslice.Windows(lead_wins)

    assert lead_wins.are_exclusive() and lead_wins.are_tight()
    return lead_wins


def plot_lead_state_cycle_counts(counts, suptitle=''):
    combs2d = [
        ('lead_ch0', 'lead_ch1'),
        #         ('x', 'lead_ch0'),
        #         ('x', 'lead_ch1'),
    ]

    f, axs = plt.subplots(constrained_layout=True, ncols=1 + len(combs2d), figsize=(1 + 2 * len(combs2d), 1.75))

    f.suptitle(suptitle)

    ax = axs[0]

    total_good = counts['lead_ch0'] + counts['lead_ch1']

    ax.hist(
        total_good - 1,
        bins=np.arange(-.5, total_good.max() + 1.5, 1),
        facecolor='k'
    )

    ax.set_ylabel('#rem periods')
    ax.set_xlabel('# switches')
    ax.set_xticks(np.arange(0, 9, 1), minor=False)
    ax.spines['bottom'].set_position(('outward', 2))

    vmax = 100

    for i, (a, b) in enumerate(combs2d):
        ax = axs[1 + i]

        h, _, _, im = ax.hist2d(

            counts[a],
            counts[b],
            bins=[np.arange(-.5, 9, 1)] * 2,
            cmap='cividis',
            norm=matplotlib.colors.Normalize(0, vmax),
        )
        ax.set_xticks(np.arange(0, 9, 1), minor=False)
        ax.set_yticks(np.arange(0, 9, 1), minor=False)
        ax.tick_params(bottom=False, left=False)

        ax.set_aspect('equal')

        f.colorbar(im, ax=ax, aspect=70).set_label('#rem periods')

        ax.set_ylabel(f'#periods "{a}"'.replace('_', ' '), color=splot.COLORS[a])
        ax.set_xlabel(f'#periods "{b}"'.replace('_', ' '), color=splot.COLORS[b])

    return axs


def smooth_lead_state_detour(lead_state_raw):
    lead_wins_raw = timeslice.Windows.build_from_contiguous_values(lead_state_raw)

    lead_wins_0 = None
    lead_wins = lead_wins_raw.copy()

    while lead_wins_0 is None or len(lead_wins_0) != len(lead_wins):
        lead_wins_0 = lead_wins
        lead_wins = lead_wins.merge_sandwiched(cat='x', max_length=15_000)
        lead_wins = lead_wins.merge_sandwiched(cat=['lead_ch0', 'lead_ch1'], max_length=5_000)

    return lead_wins.generate_cat(lead_state_raw.index)


class Cycles:
    """
    Represent a 2-level hyerarchy of windows where the
    first level represents cycles of rem-swsget_rem_wins
    and the second level subdivides those cycles into leadership periods.
    """

    def __init__(self, lead_wins):
        self.lead_wins = lead_wins
        assert 'cycle_idx' in self.lead_wins.wins.columns

    def shift(self, value):
        return self.__class__(
            self.lead_wins.shift(value)
        )

    def sel_mask_cycle(self, mask):
        return self.__class__(
            self.lead_wins.sel_mask(
                self.lead_wins['cycle_idx'].map(mask)
            )
        )

    def align_to_rem_center(self):
        rem_centers = self.get_rem_center()
        return self.align_to(rem_centers)

    def align_to(self, rem_times):

        lead_wins_shifted = self.lead_wins.shift(
            -1 * self.lead_wins['cycle_idx'].map(rem_times)
        )
        return self.__class__(
            lead_wins_shifted,
        )

    def get_period_counts_per_cycle(self) -> pd.DataFrame:
        """
        Number of periods in each cycle. Result looks like:

            cat        lead_ch0  lead_ch1  sws  x
            cycle_idx
            0                 3         2    2  4
            1                 3         2    2  4
        """
        return self.lead_wins.groupby(['cycle_idx', 'cat']).size().unstack('cat', fill_value=0)

    def get_switch_count_per_cycle(self) -> pd.Series:
        return self.get_period_counts_per_cycle()[['lead_ch0', 'lead_ch1']].sum(axis=1) - 1

    def get_period_lengths_per_cycle(self) -> pd.DataFrame:
        df = self.lead_wins.wins.copy()
        df['length'] = self.lead_wins.lengths()
        return df.groupby(['cat', 'cycle_idx'])['length'].sum().unstack('cat', fill_value=0)

    def get_period_lengths_relative(self, valid_cats=('lead_ch0', 'lead_ch1')) -> pd.Series:
        lengths = self.get_period_lengths_per_cycle()
        lengths = lengths.loc[:, list(valid_cats)]
        total = lengths.sum(axis=1)
        return (lengths.T / total).T.dropna()

    def take_largest_by_cat(self):
        df = self.lead_wins.wins.copy()
        df['length'] = self.lead_wins.lengths()

        largest = df.groupby(['cat', 'cycle_idx'])['length'].idxmax()
        assert largest.is_unique

        return self.__class__(
            self.lead_wins.sel_mask(largest.sort_values().values)
        )

    def get_center_largest(self) -> pd.DataFrame:
        largest = self.take_largest_by_cat()
        assert np.all(largest.lead_wins.groupby(['cycle_idx', 'cat']).size() <= 1)
        return largest.lead_wins.groupby(['cycle_idx', 'cat'])['ref'].first().unstack('cat')

    def get_rem_wins(self):
        rems = pd.DataFrame({
            'start': self.lead_wins.sel(cat='sws', but=True).groupby('cycle_idx')['start'].min(),
            'stop': self.lead_wins.sel(cat='sws', but=True).groupby('cycle_idx')['stop'].max(),
        })

        rems['ref'] = rems.mean(axis=1)

        return timeslice.Windows(rems)

    def get_rem_center(self) -> pd.Series:
        return self.get_rem_wins().mid()

    def plot_cycles_stack(self, ax, cycle_idcs=None, height=.8):
        grouped = self.lead_wins.groupby('cycle_idx')

        if cycle_idcs is None:
            cycle_idcs = np.sort(self.lead_wins['cycle_idx'].unique())

        for k, cycle_idx in enumerate(tqdm(cycle_idcs)):
            wins = grouped.get_group(cycle_idx)
            y = k

            splot.plot_wins_fill(
                ax,
                wins,
                ymin=y - .5 * height,
                ymax=y + .5 * height,
                transform=ax.transData,
                alpha=.5,
            )
