"""
Plotting code for beta phase analysis.
"""

import matplotlib.colors
import matplotlib.patheffects
import matplotlib.ticker
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from nocte import plot as splot
from nocte.timeslice import Windows
from nocte.timeslice import ms

PHASE_CAT_COLORS = {
    'early sws': 'C0',
    'late sws': 'C2',
    'early rem': 'C1',
    'late rem': 'C3',
}


def plot_background_phase(ax, which='y'):
    prb_rem = pd.Series(
        np.sin(np.linspace(.5, 1.5, 101) * np.pi * 2) > 0,
        index=np.linspace(0, 1, 101),
    )

    lim = ax.get_ylim() if which == 'y' else ax.get_xlim()
    yvals = np.linspace(*lim, 1001)

    gradient = np.interp(yvals % 1, prb_rem.index, prb_rem.values)

    is_rem = pd.Series(gradient >= 0.5, index=yvals)

    wins = Windows.build_from_contiguous_values(is_rem)

    wins['cat'] = wins['cat'].map({True: 'rem', False: 'sws'})

    splot.plot_wins_fill(
        ax,
        wins,
        which='x',
        window_colors=dict(
            rem='w',
            sws='xkcd:silver',
        )
    )


def plot_scatter_reflected(ax, x, y, colors, xlim=(-.5, +1.5), ylim=(-1, +1), s=50):
    df = pd.DataFrame(dict(x=x, y=y))

    reflected = get_circular_scatter(df, xlim, ylim, include_zero=False)

    ax.scatter(
        reflected['x'],
        reflected['y'],
        facecolor='xkcd:grey',
        edgecolor='w',
        linewidth=.25,
        alpha=.25,
        s=s,
    )

    ax.scatter(
        df['x'],
        df['y'],
        facecolor=colors,
        edgecolor='w',
        linewidth=.25,
        alpha=.5,
        s=s,
        clip_on=False,
    )


def get_circular_scatter(df, xlim, ylim, include_zero):
    reflected = [
        df + np.array([-1, -1]),
        df + np.array([-1, 0]),
        df + np.array([-1, +1]),
        df + np.array([0, -1]),
        df + np.array([0, +1]),
        df + np.array([+1, -1]),
        df + np.array([+1, 0]),
        df + np.array([+1, +1]),
    ]

    if include_zero:
        reflected.append(df)

    reflected = pd.concat(reflected, ignore_index=True)

    valid = reflected['x'].between(*xlim) & reflected['y'].between(*ylim)

    reflected = reflected[valid]

    return reflected


def plot_phase_response(
        ax, x, y, colors,
        exp_names=None,
        xlim=(-.5, +1.5), ylim=(-1, +1),
        zero_bottom_spine=True,
        xlabel='phase at pulse end',
        ylabel='phase shift',
        s=150,
):
    if exp_names is not None:
        x = x[exp_names]
        y = y[exp_names]
        colors = colors[exp_names]

    plot_scatter_reflected(
        ax, x, y,
        colors=colors,
        xlim=xlim,
        ylim=ylim,
        s=s,
    )

    ax.set(
        aspect='equal',
        xlim=xlim,
        ylim=ylim,
        xlabel=xlabel.replace('_', ' '),
        ylabel=ylabel.replace('_', ' '),
    )

    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=.5))
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(base=.5))

    if zero_bottom_spine:
        ax.axhline(0, color='k')

    ax.spines['bottom'].set_visible(False)


def plot_beta_spread_single_exp(
        beta,
        y_space=1.75,
        left_fill=False,
        right_fill=False,
        max_fill=True,
        summary=pd.DataFrame.median,
        suptitle=''
):
    f, axs = plt.subplots(nrows=4, gridspec_kw=dict(height_ratios=[8, 1, 1, 1]), sharex='all', figsize=(4, 5))

    f.suptitle(suptitle)

    for i, (_, traces) in enumerate(beta.iter_grouped('win_idx')):

        ax = axs[0]

        for k, which in traces['side'].items():
            trace = traces.get(k)

            color = splot.COLORS[which]

            if (which == 'left' and left_fill) or (which == 'right' and right_fill):
                ax.fill_between(
                    trace.index,
                    np.ones(len(trace)) * i * y_space,
                    np.ones(len(trace)) * i * y_space + trace,
                    trace.values,
                    facecolor=color,
                    alpha=.75,
                )
            else:
                ax.plot(
                    trace + i * y_space,
                    color=color,
                )

        if max_fill:
            trace = traces.max(axis=1)
            ax.fill_between(
                trace.index,
                np.ones(len(trace)) * i * y_space,
                np.ones(len(trace)) * i * y_space + trace,
                trace.values,
                facecolor='k',
                alpha=.75,
            )

    for i, which in enumerate(['left', 'right']):
        ax = axs[1 + i]

        traces = beta.sel(side=which)

        ax.plot(
            traces.traces,
            color=splot.COLORS[which],
            linewidth=.25,
            alpha=.25,
            clip_on=False,
        )

        ax.plot(
            summary(traces.traces, axis=1),
            color=splot.COLORS[which],
            clip_on=False,
        )

        ax.set_ylim(-.25, 2)

    ax = axs[-1]
    traces = beta.groupby_max(['exp_name', 'win_idx']).normalize_by_quantiles()

    ax.plot(
        traces.traces,
        color='k',
        linewidth=.25,
        alpha=.25,
        clip_on=False,
    )

    ax.plot(
        summary(traces.traces, axis=1),
        color='k',
        clip_on=False,
    )

    ax.set_ylim(-.25, 2)

    splot.drop_spines_grid(axs, left_edge=True)
    for ax in axs:
        splot.add_yscale_bar(ax)

        ax.axvline(0, color=splot.COLORS['pulse'], linestyle='--', linewidth=1)

    ax = axs[-1]

    splot.set_time_ticks(ax, scale='minutes')

    return f


CAP_COLORS = {
    'no': 'xkcd:grey',
    'bilat': 'k',
    'left': splot.COLORS['left'],
    'right': splot.COLORS['right'],
    'seeing': splot.COLORS['left'],
    'blind': splot.COLORS['right'],
    'sws': splot.COLORS['sws'],
}
IQR = [.25, .75]


def plot_circ_line(ax, slope, intercept=.75, split=0.25, range_x=(-.5, 1.5), color='k', linestyle='--'):
    line_x = np.linspace(*range_x, 101)

    line_y = slope * line_x + intercept

    mask = line_x < split
    line_y[mask] -= 1

    ax.plot(
        line_x,
        line_y,
        color=color,
        linestyle=linestyle,
        zorder=1e6
    )


def plot_prc_single(ax, samples, class_by, line='diag'):
    plot_phase_response(
        ax,
        samples[class_by],
        samples['phase_diff'],
        colors=samples[f'{class_by}_cat'].map(PHASE_CAT_COLORS),
        zero_bottom_spine=False,
    )
    ax.set(xlabel='phase at pulse')

    if line == 'diag':
        params = dict(slope=-1, intercept=.75, split=.25)
    else:
        params = dict(slope=0, intercept=0, split=-np.inf)

    plot_circ_line(
        ax,
        **params,
    )

    ax.set(
        xlabel=r'$\varphi$ at pulse end',
        ylabel=r'$\Delta \varphi$'
    )
    ax.xaxis.set_label_position('top')

    splot.add_desc(ax, f'n={len(samples.index):,g}', loc='upper right', bkg_color='none')


def plot_phases_grouped_single(ax, phase_detailed_cut, class_by, shaded):
    phase_detailed_cut = phase_detailed_cut.shuffle()

    if not shaded:
        for k, trace in phase_detailed_cut.traces.items():
            phase_cat = phase_detailed_cut.loc[k, class_by]

            ax.plot(
                trace,
                color=PHASE_CAT_COLORS[phase_cat],
                linewidth=.25,
                alpha=.25,
            )

    for i, phase_cat in enumerate(['late rem', 'early rem', 'late sws', 'early sws']):
        traces = phase_detailed_cut.sel(**{class_by: phase_cat})

        color = PHASE_CAT_COLORS[phase_cat]

        if shaded:
            summary = traces.median(axis=1)
            low = traces.quantile(.05, axis=1)
            high = traces.quantile(.95, axis=1)

            ax.fill_between(
                low.index,
                low,
                high,
                facecolor=color,
                alpha=.5,
                edgecolor='none',
            )

        else:
            summary = traces.mean(axis=1)

        ax.plot(
            summary,
            color=color,
            linewidth=2,
            alpha=1,
            label=f'{phase_cat} n={len(traces.index):,g}',
            zorder=1e6,
        )

    splot.set_time_ticks(ax, scale='minutes', major=ms(minutes=1))
    ax.set(ylabel='phase')

    ax.set(ylim=(-1.25, +1.75), xlim=(ms(minutes=-2), ms(minutes=+4)))
    ax.legend(loc='lower right', fontsize=4)

    plot_background_phase(ax)

    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax.axvline(0, color=splot.COLORS['pulse'], linestyle='--', zorder=1e6)
    for sp in ax.spines.values():
        sp.set_visible(True)
    ax.set(
        xlabel='Time (min)',
        ylabel=r'Phase ($\varphi$)',
        aspect=ms(seconds=120),
    )


def plot_betas_grouped_single(
        ax,
        beta_detailed_cut,
        class_by,
        colors=None,
        show_labels=True,
        shaded=False,
        n_label_offset=0.,
        clip_on=True,
):
    if colors is None:
        colors = PHASE_CAT_COLORS

    beta_detailed_cut = beta_detailed_cut.shuffle()

    y_spacing = 1.5

    for i, phase_cat in enumerate(['late rem', 'early rem', 'late sws', 'early sws']):
        traces = beta_detailed_cut.sel(**{class_by: phase_cat})

        color = colors[phase_cat]

        y_offset = i * y_spacing

        if shaded == False:
            ax.plot(
                traces.traces + y_offset,
                color=color,
                linewidth=.25,
                alpha=.125,
                clip_on=clip_on,
            )
        else:
            iqr = IQR

            if shaded != True:
                iqr = shaded

            low = traces.quantile(iqr[0], axis=1)
            high = traces.quantile(iqr[1], axis=1)

            ax.fill_between(
                low.index,
                low + y_offset,
                high + y_offset,
                facecolor=color,
                alpha=.25,
                edgecolor='none',
                clip_on=clip_on,
            )

        ax.plot(
            traces.median(axis=1) + y_offset,
            color=color,
            linewidth=1,
            alpha=1,
            label=f'{phase_cat} n={len(traces.index):,g}',
            zorder=1e5,
            clip_on=clip_on,
        )

        if show_labels:
            ax.text(
                0,
                y_offset + .5 * y_spacing,
                phase_cat.replace('_', ' ') + '\n',
                ha='left',
                va='bottom',
                transform=ax.get_yaxis_transform(),
                zorder=1e6,
                fontsize=6,
                color=color,
            )
            ax.text(
                1 + n_label_offset,
                y_offset + .5 * y_spacing,
                f'n={len(traces.index):,g}\n',
                ha='right',
                va='bottom',
                transform=ax.get_yaxis_transform(),
                zorder=1e6,
                fontsize=6,
                color=color,
            )

    splot.set_time_ticks(ax, scale='minutes', label='Time (min)')

    ax.set(
        ylim=(-.1, y_spacing * 4),
        xlim=beta_detailed_cut.get_rel_win(),
    )
    # ax.set(ylim=(-1.5, +2), xlim=(ms(minutes=-2), ms(minutes=+4)))
    #     ax.legend(loc='lower right')

    splot.drop_spine(ax, 'y')
    splot.add_yscale_bar(ax)

    ax.axvline(0, color=splot.COLORS['pulse'], linestyle='--')


def plot_phase_dependency(beta_comb_cut, phase_comb_cut, class_by='phase_single_cat'):
    beta_comb_cut = beta_comb_cut.crop((ms(minutes=-4), ms(minutes=5)))

    f, axs = plt.subplots(ncols=3, figsize=(7, 2.5))

    ax = axs[0]
    plot_phases_grouped_single(
        ax,
        phase_comb_cut.sel(seeing='seeing'),
        class_by=class_by,
        shaded=False,
    )
    ax.set_title(f'seeing claustrum', color=CAP_COLORS['seeing'])

    ax = axs[1]
    plot_phases_grouped_single(
        ax,
        phase_comb_cut.sel(seeing='blind'),
        class_by=class_by,
        shaded=False,
    )
    ax.set_title(f'blind claustrum', color=CAP_COLORS['blind'])

    ax = axs[2]
    plot_betas_grouped_single(
        ax,
        beta_comb_cut.sel(seeing='blind'),
        class_by=class_by,
        shaded=True,
        colors={
            'late sws': 'xkcd:dark grey',
            'early sws': 'xkcd:dark grey',
            'late rem': 'xkcd:dark grey',
            'early rem': 'xkcd:dark grey',
        },
        show_labels=True,
    )

    plot_betas_grouped_single(
        ax,
        beta_comb_cut.sel(seeing='seeing'),
        class_by=class_by,
        shaded=True,
        show_labels=True,
        n_label_offset=-.25,
    )

    return f
