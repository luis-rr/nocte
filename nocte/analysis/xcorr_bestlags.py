"""
Code to extract, analyse and plot the lags that yield the highest x-corr
"""
import logging
from pathlib import Path

import matplotlib.cm
import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nocte import plot as splot
from nocte.datadict import DataDict
from nocte.stacks import Stack
from nocte.timeslice import ms


class XCorrBestLags:

    def __init__(self, reg):
        self.reg: pd.DataFrame = reg
        assert self.reg['time'].is_unique

    def sel_mask(self, mask):
        return self.__class__(self.reg.loc[mask])

    @classmethod
    def from_xcorr(cls, xcorr: Stack):
        """collect the points of maximum xcorr"""
        lags = xcorr.idxmax('lag').data.to_series()
        return cls.from_xcorr_lags(xcorr, lags)

    @classmethod
    def from_xcorr_lags(cls, xcorr: Stack, lags: pd.Series):
        """collect values given the desired lag for every time point"""
        values = xcorr.sel_pairs('time', 'lag', lags.dropna()).data.to_series()

        df = pd.DataFrame.from_dict({
            'lag': lags.reindex(xcorr.coords['time']),
            'xcorr': values.reindex(xcorr.coords['time']),
        })

        df.reset_index(inplace=True)

        return cls(df)

    def __repr__(self):
        return self.reg.__repr__()

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.reg._repr_html_()

    def __getitem__(self, item):
        return self.reg.__getitem__(item)

    def plot_hist2d(self, c0='time', c1='lag', ax=None, density=True, show_cbar=True, cmap='cividis', **kwargs):
        if ax is None:
            f, ax = plt.subplots(constrained_layout=True)

        bins = []
        for c in [c0, c1]:
            if c == 'lag':
                bins.append(
                    np.arange(self.reg['lag'].min() - .5, self.reg['lag'].max() + 1, 1)
                )

            elif c == 'time':
                bins.append(
                    np.linspace(self.reg['time'].min(), self.reg['time'].max(), 301)
                )
            elif c == 'xcorr':
                bins.append(
                    np.linspace(self.reg[c].min(), self.reg[c].max(), 101)
                )

        # noinspection PyTypeChecker
        h, xedges, yedges, im = ax.hist2d(
            self.reg[c0],
            self.reg[c1],
            bins=bins,
            density=density,
            cmap=cmap,
            **kwargs,
        )

        ax.set(xlabel=c0, ylabel=c1)

        if c0 == 'time':
            splot.set_time_ticks(ax)

        elif c0 == 'lag':
            splot.set_time_ticks(ax, which='x', major=20)

        if c1 == 'lag':
            splot.set_time_ticks(ax, which='y', major=20)

        if show_cbar:
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_label('density' if density else '#samples')

    def get_hist1d_lags(self, bins=None):
        if bins is None:
            bins = np.arange(self.reg['lag'].min() - .5, self.reg['lag'].max() + 1, 1)

        if self.reg['lag'].min() < np.min(bins):
            logging.warning(f'Lags below lowest bin')

        if self.reg['lag'].max() > np.max(bins):
            logging.warning(f'Lags above highest bin')

        h, edges = np.histogram(self.reg['lag'], bins=bins)

        # noinspection PyUnresolvedReferences
        return pd.Series(h, index=pd.IntervalIndex.from_breaks(edges))

    def plot_hist1d_lags(self, ax=None, lag_bins=None, orientation='vertical', facecolor='k', **kwargs):
        if ax is None:
            f, ax = plt.subplots(constrained_layout=True)

        if lag_bins is None:
            lag_bins = np.arange(self.reg['lag'].min() - .5, self.reg['lag'].max() + 1, 1)

        # noinspection PyTypeChecker
        ax.hist(
            self.reg['lag'],
            lag_bins,
            orientation=orientation,
            facecolor=facecolor,
            **kwargs,
        )

        if orientation == 'vertical':
            ax.set_xlabel('lag (ms)')
            ax.set_ylabel('#samples')

        else:
            ax.set_ylabel('lag (ms)')
            ax.set_xlabel('#samples')

    def plot_sel_min_xcorr(self, vmin, suptitle=''):
        f, axs = plt.subplots(nrows=3, sharex='all', constrained_layout=True, figsize=(3, 3))
        f.suptitle(suptitle)

        for ax in axs:
            ax.spines['left'].set_position(('outward', 2))
            ax.spines['bottom'].set_position(('outward', 2))

        ax = axs[0]
        self.sel_xcorr_range(vmin=vmin).plot_hist1d_lags(ax=ax, facecolor='r', clip_on=False)
        ax.set_xlabel('')

        ax = axs[1]
        self.sel_xcorr_range(vmax=vmin).plot_hist1d_lags(ax=ax, facecolor='k', clip_on=False)
        ax.set_xlabel('')

        ax = axs[2]
        self.plot_hist2d('lag', 'xcorr', ax=ax, norm=matplotlib.colors.LogNorm(), clip_on=False)
        ax.axhline(vmin, color='r', linewidth=.5, linestyle='--')

    def clip_extreme_lags(self):
        """
        Remove max and min lags.
        These lags are overrepresented when lags are extracted by max corr
        using a lag range that is too short to capture auto-corr.
        """
        vmax = self.reg['lag'].max()
        vmin = self.reg['lag'].min()
        mask = ~self.reg['lag'].isin([vmax, vmin])

        return self.__class__(
            self.reg.loc[mask]
        )

    def sel_xcorr_range(self, vmin=-np.inf, vmax=+np.inf):
        """remove lags associated with too low x-corr"""
        mask = (vmin <= self.reg['xcorr']) & (self.reg['xcorr'] <= vmax)
        return self.__class__(
            self.reg.loc[mask]
        )

    def sel_xcorr_range_q(self, qmin=0., qmax=1.):
        """Remove lags associated with too low x-corr. Thresholds are taken as quantiles"""
        vmin = self.reg['xcorr'].quantile(qmin)
        vmax = self.reg['xcorr'].quantile(qmax)

        return self.sel_xcorr_range(vmin=vmin, vmax=vmax)


def extract_best_lags_multi_exp(exp_xcorr, time_range, lag_range):
    exp_best_lags = {}

    for exp_desc, xcorr in exp_xcorr.items(pbar='best lags'):
        # allow for a lag a bit bigger as we will
        # we will drop the first and last bins because these are over-represented
        # due to a boundary effect
        extended_valid_lags = (lag_range[0] - 1, lag_range[1] + 1)

        xcorr = xcorr.sel_between(time=time_range)

        assert xcorr.coords['lag'].min() <= extended_valid_lags[0], 'lags extracted in window shorter than requested'
        assert xcorr.coords['lag'].max() >= extended_valid_lags[1], 'lags extracted in window shorter than requested'
        xcorr = xcorr.sel_between(lag=extended_valid_lags)

        best_lags = XCorrBestLags.from_xcorr(xcorr)

        exp_best_lags[exp_desc.name] = best_lags.reg

    return DataDict(
        exp_xcorr.reg.loc[exp_best_lags.keys()],
        exp_best_lags
    )


def plot_dists_best_lags(
        exp_hists: pd.DataFrame, ylabel='prob', figsize=(3, 3), suptitle=None, sharey='none',
        legend=True
):
    nrows = exp_hists.columns.get_level_values('pair').nunique()

    # noinspection PyTypeChecker
    f, axs = plt.subplots(
        nrows=nrows, sharex='all', sharey=sharey, constrained_layout=True,
        figsize=figsize, squeeze=False
    )

    axs = axs.ravel()

    x = exp_hists.index

    if isinstance(x, pd.IntervalIndex):
        # noinspection PyUnresolvedReferences
        x = exp_hists.index.mid

    pairs = exp_hists.columns.get_level_values(0).unique()

    for i, pair in enumerate(pairs):

        traces = exp_hists[pair]

        ax = axs[i]
        ax.set_title(pair)
        ax.set_ylabel(ylabel)

        for exp_name, trace in traces.items():
            ax.plot(x, trace.values, label=exp_name, color='k' if not legend else None, linewidth=.5)

        if len(traces.columns) > 1:
            mean = traces.mean(axis=1)
            ax.plot(x, mean.values, color='r' if not legend else 'k', zorder=1e6, linestyle='--' if legend else '-')

        splot.set_time_ticks(ax, major=20)

        if legend:
            ax.legend(loc='upper right', fontsize=4, ncol=1 if len(traces.columns) <= 8 else 2)

        splot.add_desc(ax, f'n={len(traces.columns):,g}', loc='upper left')

    if suptitle is not None:
        f.suptitle(suptitle)

    return axs


def compute_lags_best_xcorr(
        reg, exp_names,
        fig_desc,
        use_cache=True,
        time_range=(ms(hours=2), ms(hours=11)),
        lag_range=(-50, +50),
        sliding_win=1_000,
        suffix='',
):
    cache_path = Path(f'temp/cache_{fig_desc}.h5')

    if cache_path.exists() and use_cache:
        print('loading cached best lags')
        exp_best_lags = DataDict.from_hdf(cache_path)

    else:
        print('computing best lags')
        exp_xcorr = reg.load_exp_xcorr_combs(exp_names, sliding_win=sliding_win, suffix=suffix)
        exp_best_lags = extract_best_lags_multi_exp(
            exp_xcorr,
            time_range=time_range,
            lag_range=lag_range,
        )

        cache_path.unlink(missing_ok=True)
        exp_best_lags.to_hdf(cache_path)

    return exp_best_lags


def compute_dist_best_lags(exp_best_lags, min_xcorr_q=.75) -> pd.DataFrame:
    def extract_hist_best_lags(exp_desc, best_lags: pd.DataFrame, report=False):
        best_lags = XCorrBestLags(best_lags)

        # boundary effect makes the first and last bins over-represented
        best_lags = best_lags.clip_extreme_lags()

        if report:
            best_lags.plot_sel_min_xcorr(
                vmin=np.quantile(best_lags['xcorr'], min_xcorr_q),
                suptitle=' '.join(list(exp_desc.values)),
            )

        best_lags = best_lags.sel_xcorr_range_q(qmin=min_xcorr_q)

        return best_lags.get_hist1d_lags()

    hists = exp_best_lags.extract_df(
        extract_hist_best_lags

    )

    probs = hists / hists.sum()

    return probs
