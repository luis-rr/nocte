"""
Code to analyise the cycle of sleep states
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from tqdm.auto import tqdm as pbar

from nocte import plot as splot
from nocte import timeslice
from nocte import traces as tr
from nocte.timeslice import ms, Win


def classify_by_gm(beta, max_detours=ms(seconds=10)):
    if not isinstance(beta, pd.DataFrame):
        beta = beta.to_frame()

    gm = GaussianMixture(n_components=2)
    labels = pd.Series(gm.fit_predict(beta), index=beta.index)

    label_sorting = beta.mean(axis=1).groupby(labels).mean().sort_values(ascending=True).index
    labels = labels.replace(dict(zip(label_sorting, ['sws', 'rem'])))

    if max_detours is not None:
        rem_wins = timeslice.Windows.build_from_contiguous_values(labels, include_right=False)
        rem_wins = rem_wins.merge_sandwiched(max_length=max_detours)
        labels = rem_wins.generate_cat(beta.index)

        missing = labels.isna()
        if missing.any():
            assert missing.value_counts()[True] == 1
            labels.ffill(inplace=True)

        assert not labels.isna().any()

    return labels


def classify_by_gm_log10(beta, max_detours=ms(seconds=10)):
    beta_log = np.log10(beta)
    assert not beta_log.isna().any()
    return classify_by_gm(beta_log, max_detours=max_detours)


def extract_rem_wins_multi(exp_beta: tr.Traces, key='exp_name', **kwargs):
    return {
        (k if key is None else exp_beta.loc[k, key]): extract_rem_wins(beta, **kwargs)
        for k, beta in pbar(exp_beta.traces.items(), total=len(exp_beta.index), desc='rem wins')
    }


def extract_rem_wins(beta, **kwargs):
    # We use log10 which means we don't want anything 0 or less
    # These are rare cases and beta is continuous and smooth, so just fill-fwd
    beta = beta.copy()
    beta.dropna(inplace=True)
    beta[beta <= 0] = np.nan
    beta.ffill(inplace=True)
    beta.bfill(inplace=True)
    assert not beta.isna().any()

    return timeslice.Windows.build_from_contiguous_values(
        classify_by_gm_log10(beta, **kwargs),
        include_right=False,
    )


def estimate_interval(acorrs_baselines: tr.Traces, between=Win(ms(minutes=1), ms(minutes=3))):
    return acorrs_baselines.crop(between).idxmax(axis=0)


def plot_estimated_intervals(beta_acorrs, intervals, color='k', axs=None):

    if axs is None:
        f, axs = plt.subplots(nrows=2, sharex='all', figsize=(2.5, 2.5))

    zoom_win = Win(ms(minutes=-1), ms(minutes=4))

    ax = axs[0]

    ax.plot(
        beta_acorrs.traces.loc[zoom_win.to_slice_ms()],
        zorder=1,
        linewidth=.5,
        color=color,
        alpha=.125,
    )

    ax.scatter(
        intervals,
        [beta_acorrs.tloc[i, k] for k, i in intervals.items()],
        zorder=1e6,
        facecolor=color,
        edgecolor='w',
        linewidth=.25,
        alpha=.75,
        clip_on=False,
    )

    ax.axhline(0, color=color)

    ax.set(ylabel='corr.')

    ax = axs[1]

    ax.hist(intervals, bins=zoom_win.arange(ms(seconds=10)), facecolor=color)

    splot.set_time_ticks(ax, scale='minutes', major=ms(minutes=1), minor=ms(seconds=10))

    ax.set(ylabel='count')

    stats = intervals.describe()

    q0 = stats['25%']
    q1 = stats['75%']
    mean = stats['mean']
    median = stats['50%']
    std = stats['std']
    count = stats['count']

    n_animals = beta_acorrs['animal'].nunique()

    desc = (
        f'mean std:\n{mean / ms(seconds=1):.1f}, {std / ms(seconds=1):.1f}\n\n'
        f'median [.25,.75]:\n{median / ms(seconds=1):.1f} [{q0 / ms(seconds=1):.1f}, {q1 / ms(seconds=1):.1f}]\n\n'
        f'{count:,g} rec\n'
        f'{n_animals:,g} animals\n'
    )

    splot.add_desc(ax, desc, loc='upper left', fontsize=6, bkg_color='none')

    ax = axs[0]

    ax.spines['bottom'].set_position(('data', 0))

    for ax in axs:
        for tick in ax.xaxis.get_major_ticks():
            tick.set_zorder(1e9)

        for tick in ax.xaxis.get_minor_ticks():
            tick.set_zorder(1e6)

    return axs[0].figure


def extract_rem_traces(analysis_windows, exp_rem_wins):
    rem_traces = {}

    for k, win, props in analysis_windows.iter_wins_items(show_pbar=True):
        rem_wins = exp_rem_wins[props['exp_name']].crop_to_main(win).shift(-props['ref'])

        rem_traces[k] = rem_wins.generate_cat_contiguous(100)

    return tr.Traces(
        analysis_windows.wins,
        pd.DataFrame(rem_traces)
    )
