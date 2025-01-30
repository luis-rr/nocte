"""
General utility code to plot traces and spike trains
"""
import colorsys
import pathlib

import matplotlib.cm
import matplotlib.colors
import matplotlib.gridspec
import matplotlib.patches
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm.auto import tqdm

from nocte import timeslice
from nocte.stacks import Stack
from nocte.timeslice import ms

COLORS_SIDE = dict(
    left='#0053A3',  # blue
    right='#FF0000',  # red
)

COLORS_CHANNEL = dict(
    ch0=COLORS_SIDE['left'],  # most common case
    ch1=COLORS_SIDE['right'],  # most common case
    ch2='#44AF69',  # green
    ch3='#F5C000',  # yellow
)

COLORS = {
    'left': COLORS_SIDE['left'],  # most common case
    'right': COLORS_SIDE['right'],  # most common case

    'lead_ch1': COLORS_CHANNEL['ch1'],
    'lead_ch0': COLORS_CHANNEL['ch0'],
    'is_lead_ch1': COLORS_CHANNEL['ch1'],
    'is_lead_ch0': COLORS_CHANNEL['ch0'],
    'beta_ch1': COLORS_CHANNEL['ch1'],
    'beta_ch0': COLORS_CHANNEL['ch0'],
    'beta_max': '#663C00',
    'lead_diff': '#AE76A6',
    'lead_score': '#AE76A6',

    'ch0': COLORS_CHANNEL['ch0'],
    'ch0_light': '#70BAFF',
    'ch0_dark': '#003566',
    'ch1': COLORS_CHANNEL['ch1'],
    'ch1_light': '#FFADAD',
    'ch1_dark': '#8F0000',
    'ch2': COLORS_CHANNEL['ch2'],  # green
    'ch2_light': '#98D7AE',
    'ch2_dark': '#225835',
    'ch3': COLORS_CHANNEL['ch3'],  # yellow
    'ch3_light': '#FFD747',
    'ch3_dark': '#7A6000',
    'none': 'xkcd:charcoal',
    'syn': 'xkcd:charcoal',
    'sws': 'xkcd:silver',
    'rem': '#FF9F1C',
    'rem_light': '#FFE6C6',
    'sws_light': '#F0F1F0',

    'x': 'xkcd:marigold',

    'light off': 'k',
    'light on': 'k',
    'sleep on': 'xkcd:magenta',
    'sleep off': 'xkcd:magenta',

    'ch1_to_sws': 'xkcd:grey blue',
    'ch0_to_sws': 'xkcd:tan',
    'sws_to_ch1': COLORS_CHANNEL['ch1'],
    'sws_to_ch0': COLORS_CHANNEL['ch0'],
    'ch0_to_ch1': 'xkcd:burnt orange',
    'ch1_to_ch0': 'xkcd:royal blue',

    'on': '#ffffff',
    'on_extrap': '#aaaaaa',
    'off': '#7c7c7c',
    'off_extrap': '#595959',

    'pulse': '#F9DB00',
}

XCORR_CMAP = matplotlib.colormaps['RdGy_r']

XCORR_CMAP_SOFT = matplotlib.colors.LinearSegmentedColormap.from_list(
    'truncated_cmap',
    XCORR_CMAP(np.linspace(0.1, 0.9, 256), )
)


def _make_ch_cmaps(seq=('_light', '', '_dark')) -> dict:
    return {
        ch: matplotlib.colors.LinearSegmentedColormap.from_list(
            ch,
            [
                COLORS[f'{ch}{shade}'] if f'{ch}{shade}' in COLORS else shade
                for shade in seq
            ],
            N=256
        )
        for ch in ['ch0', 'ch1', 'ch2', 'ch3']
    }


CMAPS_CHANNEL = _make_ch_cmaps(['_light', '', '_dark'])


def set_time_ticks(
        ax,
        major=None,
        minor=None,
        which='x',
        minor_length=2,
        major_length=3,
        scale=None,
        label=None,
        tight=True,
        lim=None,
):
    """set major xticks to mark every hour and minor every 10 minutes"""
    assert which in ('x', 'y')
    axis = ax.yaxis if which == 'y' else ax.xaxis

    if lim is not None:
        if which == 'x':
            ax.set_xlim(lim)
        else:
            ax.set_ylim(lim)

    if tight is not None and lim is None:
        ax.autoscale(enable=True, axis=which, tight=tight)

    ax.tick_params(which='minor', length=minor_length)
    ax.tick_params(which='major', length=major_length)

    auto_major, auto_minor = _auto_select_tick_steps(ax, which=which)

    if minor is None:
        if major is not None:
            minor = major / 2
        else:
            minor = auto_minor

    if major is None:
        major = auto_major

    axis.set_major_locator(matplotlib.ticker.MultipleLocator(timeslice.to_ms(major)))
    axis.set_minor_locator(matplotlib.ticker.MultipleLocator(timeslice.to_ms(minor))),

    scale_factor = scale
    if scale_factor is not None:
        if isinstance(scale_factor, str):
            scale_factor = ms(**{scale_factor: 1})

        def scale_ticks(x, _):
            return f'{x / scale_factor:g}'

        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(scale_ticks))

    if label is None and isinstance(scale, str) and axis.get_label_text() == '':
        aliases = dict(
            minutes='min',
            seconds='sec',
            milliseconds='ms',
        )
        label = aliases.get(scale, scale)

    if label is not None:
        axis.set_label_text(label)


def _auto_select_tick_steps(ax, which='x') -> tuple:
    """
    Select the steps of minor and major ticks according
    to the ax current data limits, so that they match common time units.

    :param ax:
    :param which:
    :return:
    """
    lim = ax.get_xlim() if which == 'x' else ax.get_ylim()

    duration = max(lim) - min(lim)

    sections = {
        ms(milliseconds=100): (ms(milliseconds=10), ms(milliseconds=5)),
        ms(seconds=1): (ms(milliseconds=100), ms(milliseconds=50)),
        ms(seconds=10): (ms(seconds=2.5), ms(milliseconds=100)),
        ms(minutes=1): (ms(seconds=10), ms(seconds=5)),
        ms(minutes=2): (ms(seconds=30), ms(seconds=10)),
        ms(minutes=5): (ms(minutes=1), ms(seconds=20)),
        ms(minutes=10): (ms(minutes=1), ms(minutes=.5)),
        ms(minutes=15): (ms(minutes=2), ms(minutes=1)),
        ms(minutes=30): (ms(minutes=3), ms(minutes=1)),
        ms(hours=1): (ms(minutes=10), ms(minutes=5)),
        ms(hours=5): (ms(minutes=30), ms(minutes=10)),
        ms(hours=10): (ms(hours=2), ms(hours=1)),
        np.inf: (ms(hours=5), ms(hours=2.5)),
    }

    assert duration < max(sections.keys())

    for thresh, steps in sections.items():
        if duration <= thresh:
            return steps


def set_ticks_solar_time(ax, which='x'):
    assert which in ('x', 'y')
    axis = ax.yaxis if which == 'y' else ax.xaxis

    def solar_ticks(x, _):
        days = np.floor(x / ms(hours=1) / 24)
        hours = (x / ms(hours=1)) % 24

        return f'{hours:g}' + (f'\n+{days:g}d' if days > 0 else '')

    axis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(solar_ticks)
    )


def drop_spine(ax, which: str):
    """
    Hide a spine and its ticks:
        drop_spine(ax, 'bottom')

    which can be 'all'
    """
    if which == 'all':
        for which in ['bottom', 'left', 'right', 'top']:
            drop_spine(ax, which)

    if which == 'x':
        which = 'bottom'

    if which == 'y':
        which = 'left'

    ax.spines[which].set_visible(False)
    ax.tick_params(**{which: False, f'label{which}': False}, which='both')


def drop_spines_grid(
        axs,
        bottom=True, bottom_edge=False, bottom_label=True,
        left=True, left_edge=False, left_label=True,
        rows=True,
):
    if len(axs.shape) != 2:
        assert rows is not None
        if rows:
            axs = axs.reshape(-1, 1)
        else:
            axs = axs.reshape(1, -1)

    if bottom:
        for ax in axs[:(-1 if not bottom_edge else None), :].ravel():
            ax.tick_params(bottom=False, labelbottom=False, which='both')
            ax.spines['bottom'].set_visible(False)

            if bottom_label:
                ax.set(xlabel='')

    if left:
        for ax in axs[:, (1 if not left_edge else None):].ravel():
            ax.tick_params(left=False, labelleft=False, which='both')
            ax.spines['left'].set_visible(False)

            if left_label:
                ax.set(ylabel='')


def plot_wins_fill(
        ax,
        windows: timeslice.Windows,
        ymin=0,
        ymax=1,
        which='y',
        by='cat',
        window_colors=None, show_excluded=False,
        show_edges=False,
        transform=None,
        facecolor=None,
        **kwargs):
    """
    plot multiple windows as shaded fill_between
    :param show_excluded: bool. whether to automatically calculate and highlight any period not covered in the windows

    :param which:
    :param show_edges:
    :param transform:
    :param facecolor:
    :param ax:
    :param windows:
    :param ymin:
    :param ymax:
    :param by:
    :param window_colors:
    :param kwargs:
    :return:
    """

    assert which in ('x', 'y')

    if hasattr(windows, 'wins'):
        windows = windows.wins

    if transform is None:
        if which == 'y':
            transform = ax.get_xaxis_transform()
        else:
            transform = ax.get_yaxis_transform()

    assert isinstance(windows, pd.DataFrame)

    if by not in windows.columns:
        windows = windows.copy()
        windows[by] = 'baseline'

    if show_excluded:
        excluded = windows.invert_windows(start=ax.get_xlim()[0], stop=ax.get_xlim()[1])
        excluded[by] = 'excluded'
        windows = pd.concat([windows, excluded], axis=0, sort=True, ignore_index=True)

    default_kwargs = dict(
        edgecolor='none',
        linewidth=0.,
        alpha=.5,
    )
    default_kwargs.update(kwargs)
    kwargs = default_kwargs

    window_colors = get_colors_with_defaults(window_colors, windows[by])

    for cat, wins in windows.groupby(by):
        label = cat

        for _, (start, stop) in wins[['start', 'stop']].astype(float).iterrows():

            if which == 'y':
                args = [
                    [start, stop],
                    [ymin, ymin],
                    [ymax, ymax],
                ]
            else:
                args = [
                    [ymin, ymax],
                    [start, start],
                    [stop, stop],
                ]

            ax.fill_between(
                *args,
                transform=transform,
                facecolor=window_colors[cat] if facecolor is None else facecolor,
                label=label,
                **kwargs,
            )
            label = None

    if show_edges:
        edges = timeslice.Windows(windows).get_edges()
        for t in edges:
            if which == 'y':
                ax.axvline(t, ymin=ymin, ymax=ymax, linewidth=.5, color='xkcd:black')
            else:
                ax.axhline(t, ymin=ymin, ymax=ymax, linewidth=.5, color='xkcd:black')


def plot_wins_line(ax, wins, yval, by='cat', colors=None, solid_capstyle='butt', **kwargs):
    colors = get_colors_with_defaults(colors, wins[by])

    for cat in wins.wins[by].unique():
        sel_wins = wins.sel(cat=cat)

        color = colors[cat]

        if 'color' in kwargs:
            color = kwargs.pop('color')

        ax.plot(
            sel_wins.wins[['start', 'stop']].T.values,
            [yval, yval],
            color=color,
            solid_capstyle=solid_capstyle,
            **kwargs,
        )


def plot_wins_rectangle(
        ax, wins, y0=0., y1=1.,
        transform=None, clip_on=False, colors=None, by='cat',
        how='face',
        **kwargs
):
    assert how in ['face', 'edge']
    colors = get_colors_with_defaults(colors, wins[by])

    if transform is None:
        transform = ax.get_xaxis_transform()

    for win_idx, x0, x1, cat in wins.wins[['start', 'stop', by]].itertuples():

        color = colors[cat]

        full_kwargs = dict(
            width=x1 - x0,
            height=y1 - y0,
            transform=transform,
            clip_on=clip_on,
        )
        if how == 'face':
            full_kwargs['facecolor'] = color
            full_kwargs['edgecolor'] = 'none'

        else:
            full_kwargs['edgecolor'] = color
            full_kwargs['facecolor'] = 'none'

        full_kwargs = {**full_kwargs, **kwargs}

        rect = matplotlib.patches.Rectangle(
            (x0, y0),
            **full_kwargs
        )

        ax.add_patch(rect)

        ax.update_datalim([[x0, y0], [x1, y1]])


def plot_spectrogram(ax, spec, yscale='log', ylim=None, scale='minutes', shading='nearest', cmap='jet', norm=None):
    if isinstance(spec, Stack):
        assert len(spec.coords) == 2
        spec = spec.transpose('freq', 'time')

        values = spec.values,
        time = spec.coords['time']
        freq = spec.coords['freq']

    else:
        assert isinstance(spec, pd.DataFrame)

        values = spec.values
        time = spec.index
        freq = spec.columns

    if ylim is None:
        valid_freqs = freq[freq > 0]

        low = np.ceil(np.log10(valid_freqs.min()))
        high = np.floor(np.log10(valid_freqs.max()))

        ylim = (10 ** low, 10 ** high)

    im = ax.pcolormesh(
        time,
        freq,
        values.T,
        shading=shading,
        cmap=cmap,
        norm=norm,
    )

    ax.set(
        yscale=yscale,
        ylim=ylim,
        ylabel='Hz',
    )

    set_time_ticks(ax, scale=scale)

    return im


def get_symmetric_norm(
        s: Stack,
        norm_iqr=(0., 1.)
):
    """
    Build a Normalize object with 0 in the center and
    min/max defined as quantiles of a given stack object.
    """
    values = s.values.ravel()

    vrange = (
        np.nanquantile(values, norm_iqr[0]),
        np.nanquantile(values, norm_iqr[1])
    )
    vmax = np.max(np.abs(vrange))
    norm = matplotlib.colors.Normalize(-vmax, +vmax)

    return norm


def plot_stack_2d(
        ax,
        s: Stack,
        cmap='RdGy_r',
        norm=None,
        xcol='time',
        aspect='auto',
        origin='lower',
        **kwargs,
):
    """
    Plot a stack as a 2d image: time vs channel (or some other dimension like depth)
    """
    assert s.ndim == 2
    assert xcol in s.dims

    if norm is None:
        norm = get_symmetric_norm(s)

    ycol = s.get_coords_names_except(xcol)[0]

    s = s.transpose(..., xcol)

    extent = _get_stack_extent(s, xcol, ycol)

    im = ax.imshow(
        s.values,
        origin=origin,
        extent=extent,
        norm=norm,
        cmap=cmap,
        aspect=aspect,
        **kwargs,
    )

    _set_axis_label(ax, xcol, 'x')
    _set_axis_label(ax, ycol, 'y')

    return im


def _set_axis_label(ax, label, which):
    """
    Smart set the label of the axis with special cases for time.

    :param ax:
    :param label:
    :param which:
    :return:
    """
    axis = ax.yaxis if which == 'y' else ax.xaxis
    axis.set_label_text(label)


def make_axs_long_experiment(
        win_ms,
        tbin_width=timeslice.ms(hours=2),
        sharey='all',
        constrained_layout=True,
        figsize=None,
        major=timeslice.ms(minutes=10),
        minor=timeslice.ms(minutes=1),
        show_timestamp=True,
        tstart_timestamp=None,
        time_scale='minutes',
        suptitle=None,
        ylim=None,
        leftspine=True,
) -> dict:
    """
    Prepare axes to plot a ful experiment chopped up in sequential chunks as rows.
    """
    win_ms = timeslice.Win(*win_ms)
    tbin_width = timeslice.to_ms(tbin_width)
    t_edges = np.arange(win_ms.start, win_ms.stop, tbin_width)
    t_edges = np.append(t_edges, win_ms.stop)

    nrows = len(t_edges) - 1

    if tstart_timestamp is not None:
        tstart_timestamp = pd.Timestamp(tstart_timestamp)

    if figsize is None:
        figsize = (7, 5 / 9 * (nrows + 1))

    # noinspection PyTypeChecker
    f, axs = plt.subplots(
        nrows=nrows,
        squeeze=False,
        sharex='all', sharey=sharey,
        constrained_layout=constrained_layout,
        figsize=figsize,
    )

    if suptitle is not None:
        f.suptitle(suptitle)

    for i, tbin in enumerate(zip(t_edges[:-1], t_edges[1:])):

        tbin = timeslice.Win(tbin[0], tbin[1])

        ax = axs.ravel()[i]

        ax.spines['bottom'].set_visible(False)
        ax.tick_params(bottom=False, which='both')

        if show_timestamp:

            if tstart_timestamp is None:
                timestamp = (
                    f'{timeslice.strf_ms(tbin.start, plus_sign=False)}'
                    f'-{timeslice.strf_ms(tbin.stop, plus_sign=False)}'
                )

            else:
                timestamp = f'{(tstart_timestamp + timeslice.timedelta(milliseconds=tbin.start))}'

                # if i == 0:
                #     timestamp += f' ({timeslice.strf_ms(tbin.length, plus_sign=True)})'

            ax.text(
                0, 1, timestamp,
                va='bottom', ha='left', fontsize=6, zorder=1e6, transform=ax.transAxes, clip_on=False)

        if ylim is not None:
            ax.set_ylim(*ylim)

        if not leftspine:
            ax.tick_params(left=False, labelleft=False)
            ax.spines['left'].set_visible(False)

    ax = axs.ravel()[-1]
    set_time_ticks(ax, major=major, minor=minor, scale=time_scale, lim=(0, tbin_width))
    ax.tick_params(bottom=True, which='major', length=3)
    ax.tick_params(bottom=True, which='minor', length=2)
    ax.spines['bottom'].set_visible(True)

    tbins = [timeslice.Win(start, stop) for start, stop in zip(t_edges[:-1], t_edges[1:])]
    return dict(zip(tbins, axs.ravel()))


def plot_events_vline(ax, events: pd.Series, colors=None, **kwargs):
    """plot a vertical line for every event"""

    if colors is None:
        colors = {}

    colors = {**COLORS, **colors}

    for name, e in events.items():
        ax.axvline(e, color=colors.get(name, 'k'), zorder=1e4, linewidth=1, clip_on=False, **kwargs)


def plot_traces_wrapped(
        all_traces=None, events=None, wins=None, axhline=None,
        linewidth=.5, alpha=.9,
        figsize=None,
        tbin_width=timeslice.ms(hours=2.5),
        suptitle=None,
        ylim=(-.25, 1),
        major=None, minor=None,
        colors=None,
        show_timestamp=True,
        tstart_timestamp=None,
        time_scale='minutes',
        win_ms=None,
        legend=True,
):
    """plot multiple traces, events and shaded windows for long experiments as wrapped axes"""

    if colors is None:
        colors = COLORS
    else:
        colors = {**colors, **COLORS}

    if win_ms is None:
        win_ms = (all_traces.index.min(), all_traces.index.max())

    axs = make_axs_long_experiment(
        win_ms,
        tbin_width=tbin_width,
        figsize=figsize,
        show_timestamp=show_timestamp,
        tstart_timestamp=tstart_timestamp,
        time_scale=time_scale,
        major=major,
        minor=minor,
        ylim=ylim,
        leftspine=False,
    )

    if wins is not None:
        plot_wrapped_wins_fill(axs, wins, colors=colors)

    plot_wrapped_lines(axs, all_traces, colors=colors, linewidth=linewidth, alpha=alpha)

    for i, (tbin, ax) in enumerate(axs.items()):

        if axhline is not None:
            ax.plot([0, tbin.length], [axhline, axhline], color='k', linewidth=1)

        if events is not None:
            plot_events_vline(ax, events[events.between(*tbin)] - tbin.start, colors=colors)

    ax = list(axs.values())[-1]
    if legend:
        ax.legend(loc='lower right', fontsize=6)

    if suptitle is not None:
        ax.figure.suptitle(suptitle)

    return axs


def plot_wrapped_lines(
        axs,
        all_traces: pd.DataFrame,
        colors=None,
        linewidth=.5,
        **kwargs,
):
    if colors is None:
        colors = COLORS
    else:
        colors = {**COLORS, **colors}

    if isinstance(all_traces, pd.Series):
        all_traces = all_traces.to_frame()

    for i, (tbin, ax) in enumerate(axs.items()):
        for j, (name, trace) in enumerate(all_traces.items()):
            name: str

            trace = tbin.crop_df(trace, reset=True)

            ax.plot(
                trace.index,
                trace.values,
                color=colors.get(name, f'C{j}'),
                label=str(name).replace('_', ' '),
                linewidth=linewidth,
                **kwargs,
            )


def plot_wrapped_scatter(
        axs,
        series,
        **kwargs,
):
    for i, (tbin, ax) in enumerate(axs.items()):
        section = tbin.crop_df(series, reset=True)

        ax.scatter(
            section.index, section.values,
            **kwargs,
        )


def plot_wrapped_lines_highlighted(
        axs,
        all_traces: pd.DataFrame,
        wins,
        styles=None,
        **kwargs,
):
    if isinstance(all_traces, pd.Series):
        all_traces = all_traces.to_frame()

    for i, (tbin, ax) in enumerate(axs.items()):
        for j, (name, trace) in enumerate(all_traces.items()):
            name: str

            tbin: timeslice.Win

            # noinspection PyTypeChecker
            trace: pd.Series = tbin.crop_df(trace, reset=True)

            wins_sel = wins.crop_to_main(tbin, reset=True)

            plot_trace_highlighted(
                ax,
                trace=trace,
                wins=wins_sel,
                styles=styles,
                **kwargs,
            )


def plot_wrapped_wins_fill(
        axs,
        wins,
        colors=None,
        alpha=.5,
        **kwargs,
):
    if colors is None:
        colors = COLORS
    else:
        colors = {**COLORS, **colors}

    for i, (tbin, ax) in enumerate(axs.items()):
        plot_wins_fill(
            ax,
            wins.crop_to_main(tbin, reset=True),
            window_colors=colors,
            alpha=alpha,
            **kwargs,
        )


def plot_wrapped_events(
        axs,
        events,
        **kwargs,
):
    """plot multiple traces, events and shaded windows for long experiments as wrapped axes"""

    for i, (tbin, ax) in enumerate(axs.items()):
        if events is not None:
            plot_events_vline(
                ax, events[events.between(*tbin)] - tbin.start,
                **kwargs,
            )


def plot_wrapped_wins_lines(
        axs,
        wins: timeslice.Windows,
        **kwargs,
):
    for i, (tbin, ax) in enumerate(axs.items()):
        wins_sel = wins.crop_to_main(tbin, reset=True)

        plot_wins_line(
            ax, wins_sel,
            **kwargs,
        )


def plot_wrapped_wins_edges(
        axs,
        wins: timeslice.Windows,
        **kwargs,
):
    for i, (tbin, ax) in enumerate(axs.items()):
        # Note we don't want to introduce artificial edges due to the cropping
        wins_sel = wins.sel_mask(wins.is_within(tbin)).crop_to_main(tbin, reset=True)

        plot_wins_edges(
            ax, wins_sel,
            **kwargs,
        )


def plot_wins_edges(
        ax,
        wins: timeslice.Windows,
        ymin=0, ymax=1,
        transform=None,
        linewidth=.5,
        color='xkcd:black',
        skip_ends=False,
        **kwargs,
):
    if transform is None:
        transform = ax.get_xaxis_transform()

    edges = wins.get_edges()
    if skip_ends:
        edges = edges[1:-1]

    for t in edges:
        ax.plot(
            [t, t], [ymin, ymax],
            linewidth=linewidth,
            color=color,
            transform=transform,
            **kwargs,
        )


def _get_stack_extent(s, xcol, ycol):
    # we the coordinates centered on each pixel!
    # add half a bin before/after
    return (
        s.coords[xcol][0] * 1.5 - s.coords[xcol][1] * .5,
        s.coords[xcol][-1] * 1.5 - s.coords[xcol][-2] * .5,

        s.coords[ycol][0] * 1.5 - s.coords[ycol][1] * .5,
        s.coords[ycol][-1] * 1.5 - s.coords[ycol][-2] * .5,
    )


def plot_traces2d_wrapped(
        stack2d,
        events=None,
        tbin_width=ms(hours=2.5),
        xcol='time', ycol='lag',
        grid_steps=None,
        suptitle=None,
        cmap='RdGy_r',
        norm=None,
        majory=20,
        minory=10,
        scaley=None,
        figsize=(9, 5),
        major=ms(minutes=15),
        minor=ms(minutes=5),
        tstart_timestamp=None,
        time_scale='minutes',
        show_timestamp=True,
        show_colorbar=False,
        cbar_extend='both',
        interpolation='antialiased',
):
    stack2d = stack2d.transpose(..., xcol)

    axs = make_axs_long_experiment(
        stack2d.get_rel_win(),
        tbin_width=tbin_width,
        figsize=figsize,
        show_timestamp=show_timestamp,
        tstart_timestamp=tstart_timestamp,
        time_scale=time_scale,
        major=major,
        minor=minor,
        suptitle=suptitle,
    )

    if norm is None:
        vmax = np.quantile(np.abs(stack2d.values), 0.999)
        norm = matplotlib.colors.Normalize(-vmax, vmax)

    if events is None:
        events = pd.Series([], dtype=float)

    for i, (tbin, ax) in tqdm(enumerate(axs.items()), total=len(axs)):

        s = stack2d.sel_between(time=tbin).shift_coord(time=-tbin[0])

        extent = _get_stack_extent(s, xcol, ycol)

        im = ax.imshow(
            s.values,
            origin='lower',
            extent=extent,
            norm=norm,
            cmap=cmap,
            interpolation=interpolation,
            aspect='auto',
        )

        # imshow might change the xlim and
        # the last plot may be shorter, so make sure we show the full tbins
        ax.set_xlim(0, tbin_width)

        set_time_ticks(ax, which='y', major=majory, minor=minory, scale=scaley)

        es = events[events.between(*tbin)] - tbin.start
        plot_events_vline(ax, es)

        if grid_steps is not None:
            for lag in grid_steps:
                ax.axhline(lag, color='xkcd:black', linewidth=.25, alpha=.5, linestyle='--')

        ax.set_xlabel('')

        if show_colorbar and i == len(axs) - 1:
            axins = inset_axes(
                ax,
                width=.05,
                height="75%",
                loc='center right',
                borderpad=2,
            )

            ax.figure.colorbar(im, ax=ax, cax=axins, extend=cbar_extend)

            axins.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2))

    return axs


def add_desc(ax, desc, loc='upper right', bkg_color='w', bkg_edgecolor='none', fontsize=6, loc_pad=0.05, bkg_alpha=0.75, **kwargs):
    """add a small text description on the axes, usually for n=X """

    coords_options = {
        'upper right': dict(x=1 - loc_pad, y=1 - loc_pad, va='top', ha='right'),
        'upper left': dict(x=0 + loc_pad, y=1 - loc_pad, va='top', ha='left'),
        'upper center': dict(x=.5, y=1 - loc_pad, va='top', ha='center'),

        'middle right': dict(x=1 - loc_pad, y=.5, va='center', ha='right'),
        'middle left': dict(x=0 + loc_pad, y=.5, va='center', ha='left'),
        'middle center': dict(x=.5, y=.5, va='center', ha='center'),

        'lower right': dict(x=1 - loc_pad, y=0 + loc_pad, va='bottom', ha='right'),
        'lower left': dict(x=0 + loc_pad, y=0 + loc_pad, va='bottom', ha='left'),
        'lower center': dict(x=.5, y=0 + loc_pad, va='bottom', ha='center'),
    }

    for x in 'right', 'center', 'left':
        coords_options[f'bottom {x}'] = coords_options[f'lower {x}']
        coords_options[f'top {x}'] = coords_options[f'upper {x}']
        coords_options[f'center {x}'] = coords_options[f'middle {x}']

    assert loc in coords_options, f'Expected one of: {list(coords_options.keys())}. Got: {loc}'
    coords = coords_options[loc]

    default_kwargs = dict(
        s=desc,
        transform=ax.transAxes,
        fontsize=fontsize,
        zorder=1e6,
        bbox=dict(facecolor=bkg_color, alpha=bkg_alpha, edgecolor=bkg_edgecolor, linewidth=0.5),
    )

    kwargs = {
        **default_kwargs,
        **coords,
        **kwargs,
    }

    return ax.text(**kwargs)


def filter_desc(hz: tuple, decimals=None) -> str:
    if hz is None:
        return 'raw'

    low, high = hz
    low_open = low is None or np.isclose(low, 0) or np.isinf(low) or np.isnan(low)
    high_open = high is None or np.isclose(high, 0) or np.isinf(high) or np.isnan(high)

    if low_open and high_open:
        return 'raw'
    else:
        if low_open:
            if decimals is not None:
                high = np.round(high, decimals=decimals)

            return f'<{high}hz'

        elif high_open:
            if decimals is not None:
                low = np.round(low, decimals=decimals)

            return f'>{low}hz'

        else:
            if decimals is not None:
                high = np.round(high, decimals=decimals)
                low = np.round(low, decimals=decimals)

            return f'{low}-{high}hz'


def make_ax_with_marginals(figsize=(3, 2), constrained_layout=True, size_ratio=3):
    f = plt.figure(
        figsize=figsize,
        constrained_layout=constrained_layout
    )

    gs = matplotlib.gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=f,
        width_ratios=[size_ratio, 1],
        height_ratios=[1, size_ratio],
    )

    axs_dict = {
        'main': f.add_subplot(gs[1, 0]),
        'xmargin': f.add_subplot(gs[0, 0]),
        'ymargin': f.add_subplot(gs[1, 1]),
    }

    axs_dict['main'].get_shared_x_axes().join(axs_dict['main'], axs_dict['xmargin'])
    axs_dict['main'].get_shared_y_axes().join(axs_dict['main'], axs_dict['ymargin'])

    axs_dict['ymargin'].tick_params(left=False, labelleft=False)
    axs_dict['xmargin'].tick_params(bottom=False, labelbottom=False)

    return axs_dict


def make_axs_grid_with_marginals(
        nrows=1, ncols=1, figsize=(9, 5), constrained_layout=True,
        size_ratio=3,
        hspace=0,
        wspace=0,
        xmargin=True,
        ymargin=True,
        sharex=True,
        sharey=True,
        spines=False,
):
    import matplotlib.gridspec

    axs = np.empty((nrows, ncols), dtype=object)

    fig = plt.figure(
        figsize=figsize,
        constrained_layout=constrained_layout,
    )

    gs = matplotlib.gridspec.GridSpec(nrows, ncols, figure=fig)

    first = None

    for i in range(nrows):
        for j in range(ncols):

            kwargs = {
                (True, True): dict(
                    nrows=2,
                    ncols=2,
                    width_ratios=[size_ratio, 1],
                    height_ratios=[1, size_ratio],
                ),
                (True, False): dict(
                    nrows=2,
                    ncols=1,
                    height_ratios=[1, size_ratio],
                ),
                (False, True): dict(
                    nrows=1,
                    ncols=2,
                    width_ratios=[size_ratio, 1],
                ),
                (False, False): dict(
                    nrows=1,
                    ncols=1,
                ),
            }[xmargin, ymargin]

            sub_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
                **kwargs,
                subplot_spec=gs[i, j],
                hspace=hspace,
                wspace=wspace,
            )

            axs_dict: dict = {
                'main': fig.add_subplot(
                    sub_gs[1 if xmargin else 0, 0],
                    sharex=first if first is not None and sharex else None,
                    sharey=first if first is not None and sharey else None,
                ),
            }

            if first is None:
                first = axs_dict['main']

            if xmargin:
                axs_dict['xmargin'] = fig.add_subplot(sub_gs[0, 0], sharex=axs_dict['main'])
                axs_dict['xmargin'].tick_params(bottom=False, labelbottom=False, which='both')
                axs_dict['xmargin'].spines['bottom'].set_visible(spines)

            if ymargin:
                axs_dict['ymargin'] = fig.add_subplot(sub_gs[1 if xmargin else 0, 1], sharey=axs_dict['main'])
                axs_dict['ymargin'].tick_params(left=False, labelleft=False, which='both')
                axs_dict['ymargin'].spines['left'].set_visible(spines)

            axs[i, j] = axs_dict

    return fig, axs


def get_colors_with_defaults(given, states) -> dict:
    if given is None:
        given = {}

    for i, state in enumerate(np.unique(states)):
        given.setdefault(state, COLORS.get(state, f'C{i}'))

    return given


def plot_scat_with_marginals(
        samples,
        by=None,
        alpha=1,
        s=.5,
        figsize=(3, 3),
        colors=None,
        suptitle='',
        density=False,
        xlabel=None,
        ylabel=None,
        rasterized=False,
        constrained_layout=True, size_ratio=3,
        clip_on=True,
        bin_count=50,
        facecolor=None,
        **kwargs,
):
    assert samples.shape[1] == 2

    if by is None:
        by = ['none'] * len(samples)

    if not isinstance(by, pd.Series):
        by = pd.Series(np.asarray(by), index=samples.index)

    colors = get_colors_with_defaults(colors, by)

    axs = make_ax_with_marginals(
        figsize=figsize,
        constrained_layout=constrained_layout,
        size_ratio=size_ratio,
    )

    ax = axs['main']

    xcol = samples.columns[0]
    ycol = samples.columns[1]

    ax.scatter(
        samples[xcol],
        samples[ycol],
        facecolor=by.map(colors) if facecolor is None else facecolor,
        s=s,
        alpha=alpha,
        rasterized=rasterized,
        clip_on=clip_on,
        **kwargs,
    )

    for i, ch in enumerate([xcol, ycol]):

        loc = 'xmargin' if i == 0 else 'ymargin'

        bins = np.linspace(
            samples[ch].replace(-np.inf, np.nan).min(),
            samples[ch].replace(+np.inf, np.nan).max(),
            bin_count + 1,
        )

        ax = axs[loc]
        for state in by.unique():
            ax.hist(
                samples.loc[by == state, ch],
                facecolor=colors[state] if facecolor is None else facecolor,
                alpha=.5,
                density=density,
                bins=bins,
                orientation='horizontal' if loc == 'ymargin' else 'vertical',
                label=state,
            )

            if loc == 'ymargin':
                ax.spines['left'].set_position(('outward', 2))
                ax.set_xlabel('prob' if density else 'count')

            else:
                ax.spines['bottom'].set_position(('outward', 2))
                ax.set_ylabel('prob' if density else 'count')

    ax = axs['main']

    if xlabel is None:
        xlabel = xcol.replace('_', ' ')

    if ylabel is None:
        ylabel = ycol.replace('_', ' ')

    ax.set(xlabel=xlabel, ylabel=ylabel)

    ax.figure.suptitle(suptitle)

    return axs


def plot_trace_highlighted(ax, trace: pd.Series, wins: timeslice.Windows, styles: dict = None, **kwargs):
    """
    Plot a trace in sections given by some windows.
    The style of each section is determined by the category of each window
    """
    # assert wins.are_exclusive()

    if styles is None:
        styles = {}

    # noinspection PyTypeChecker
    cropped = wins.crop_df(trace, show_pbar=len(wins) > 1000, reset=None).items()

    if len(cropped) > 1000:
        cropped = tqdm(cropped, desc='plot')

    for win_idx, trace in cropped:

        style = styles.get(wins['cat'][win_idx], {})

        if isinstance(style, str):
            style = dict(color=style)

        line_style = {
            **style,
            **kwargs,
        }
        ax.plot(trace, **line_style)


def add_yscale_bar(ax, *args, **kwargs):
    add_scale_bar(
        ax, 'y', *args, **kwargs,
    )


def add_scale_bar(
        ax,
        which,
        clip_on=False,
        desc=None,
        pos=0,
        vmin=0,
        vmax=1,
        nospine=True,
        color='k',
        linewidth=2,
        zorder=1e6,
        fontsize=6,
        unit='',
        **kwargs
):
    if isinstance(pos, str):
        pos = {
            'upper': 1,
            'top': 1,
            'bottom': 0,
            'lower': 0,
            'left': 0,
            'right': 1,
        }[pos]

    if nospine:
        if which == 'y':
            ax.tick_params(
                left=False, labelleft=False,
                right=False, labelright=False,
                which='both',
            )
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.tick_params(
                bottom=False, labelbottom=False,
                top=False, labeltop=False,
                which='both',
            )
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(False)

    if which == 'y':
        transform = ax.get_yaxis_transform()
        x = [pos, pos]
        y = [vmin, vmax]
    else:
        transform = ax.get_xaxis_transform()
        y = [pos, pos]
        x = [vmin, vmax]

    ax.plot(
        x, y,
        color=color,
        linewidth=linewidth,
        clip_on=clip_on,
        transform=transform,
        solid_capstyle='butt',
        zorder=zorder,
        **kwargs,
    )

    if desc is not None:

        coord = [pos, np.mean([vmin, vmax])]

        if which == 'x':
            coord = coord[::-1]

        if isinstance(desc, bool) and desc:
            desc = f'{vmax - vmin:g}{unit}'

        ax.text(
            *coord,
            f'{desc}\n',
            clip_on=clip_on,
            rotation=90 if which == 'y' else None,
            ha='center',
            va='center',
            fontsize=fontsize,
            transform=transform,
        )


def darken_color(original_color_name, darken_factor):
    original_color = matplotlib.colors.to_rgb(original_color_name)

    hsl_color = colorsys.rgb_to_hls(*original_color)

    darkened_hsl_color = hsl_color[0], hsl_color[1], np.clip(hsl_color[2] * darken_factor, 0, 1)

    darkened_rgb_color = colorsys.hls_to_rgb(*darkened_hsl_color)

    return darkened_rgb_color


def plot_racorr(
        ax,
        acorr,
        cmap=XCORR_CMAP_SOFT,
        norm=None,
        aspect='auto',
        interpolation='none',
        yscale=None,
        xscale=None,
        **kwargs
):
    if norm is None:
        vmax = 1
        norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)

    # noinspection PyTypeChecker
    im = plot_df_as_im(
        ax,
        acorr,
        aspect=aspect,
        interpolation=interpolation,
        cmap=cmap,
        norm=norm,
        **kwargs,
    )

    if xscale is not None:
        set_time_ticks(ax, scale=xscale, which='x')

    if yscale is not None:
        set_time_ticks(ax, scale=yscale, which='y')

    return im


def plot_df_as_im(
        ax,
        df: pd.DataFrame,
        cmap='viridis',
        norm=None,
        aspect='auto',
        interpolation='none',
        **kwargs
):
    """
    It will plot index (typically time) as X and columns (features) as Y
    """
    x_extent = df.index[[0, -1]] + np.diff(df.index)[[0, -1]] * np.array([-.5, +.5])
    y_extent = df.columns[[0, -1]] + np.diff(df.columns)[[0, -1]] * np.array([-.5, +.5])

    im = ax.imshow(
        df.T,
        extent=(*x_extent, *y_extent),
        aspect=aspect,
        interpolation=interpolation,
        cmap=cmap,
        norm=norm,
        origin='lower',
        **kwargs,
    )

    ax.set(
        xlabel=df.index.name,
        ylabel=df.columns.name,
    )

    return im


def plot_light_protocol_bar(
        ax, light_wins, y0=1, y1=1.05,
        divisor=None,
        transform=None,
        edgecolor='k',
        linewidth=.5,
        clip_on=False,
):
    if transform is None:
        transform = ax.get_xaxis_transform()

    plot_wins_rectangle(
        ax,
        light_wins,
        y0=y0,
        y1=y1,
        clip_on=clip_on,
        colors=COLORS,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=transform,
    )

    if divisor is not None:
        for t in light_wins.get_edges()[1:-1]:
            ax.axvline(t, color=divisor, linestyle='--', linewidth=.25)


def p_value_stars_level(p_value: float) -> int:
    if p_value < 0.001:
        return 3

    elif p_value < 0.01:
        return 2

    elif p_value < 0.05:
        return 1

    else:
        return 0


def p_value_stars(p_value: float) -> str:
    level_string = [
        'n.s.',
        '*',
        '**',
        '***',
    ]

    return level_string[p_value_stars_level(p_value)]


def savefig(f, name):
    figs_path = pathlib.Path('/gpfs/laur/data/fenkl/from_luis')

    name = name.replace(' ', '_')
    name = name.replace('\n', '_')

    full_path = figs_path / f'{name}.pdf'
    full_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Saving: {full_path}')
    f.savefig(full_path, dpi=600)


def plot_pulse_shade(
        ax,
        pulse_win,
        ymin=0,
        ymax=1,
        transform=None,
):
    if transform is None:
        transform = ax.get_xaxis_transform()

    ax.fill_between(
        pulse_win,
        [ymin] * 2,
        [ymax] * 2,
        facecolor='#F9DB00',
        transform=transform,
    )


def format_p_value(p, min_p_digits=4):
    """Turn a small p value into a string showing at least one significant digit and a minimum number of digits"""
    assert 0 <= p <= 1, f'{p}'
    p_digits = int(np.ceil(-np.log10(p)))
    p_text = '{p:.' + str(max(p_digits + 1, min_p_digits)) + 'f}'
    p_text = p_text.format(p=p).rstrip('0')

    return p_text


def plot_test(
        ax,
        p,
        stat_name,
        stat,
        ns,
        baseline_x=0, effect_x=1,
        y=.9,
        transform=None,
        sig_color=None,
        ticks=None,
        desc=None,
        tick_height=.025,
):
    if transform is None:
        transform = ax.get_xaxis_transform()

    text = ''

    p_text = format_p_value(p)

    text += p_value_stars(p)
    text += f'\n{stat_name}={stat} p={p_text}'

    if len(ns) > 1:
        text += f'\n' + ', '.join([f'n{i}={n:,g}' for i, n in enumerate(ns)])
    else:
        text += f'\n' + f'n={ns[0]:,g}'

    if desc is not None:
        text += f'{desc}'

    color = 'k'

    if sig_color is not None and p_value_stars_level(p) > 0:
        color = sig_color

    ax.plot(
        [baseline_x, baseline_x, effect_x, effect_x],
        [y - tick_height, y, y, y - tick_height],
        transform=transform,
        linewidth=.5,
        color=color,
    )

    ax.text(
        baseline_x,
        y,
        text,
        ha='left',
        va='bottom',
        fontsize=5,
        transform=transform,
        color=color,
    )

    if ticks is not None:
        ax.set(
            xticks=[baseline_x, effect_x],
            xticklabels=ticks,
        )


def wilcoxon_test(
        ax,
        baseline, effect,
        alternative='two-sided',
        effect_size=False,
        **kwargs,
):
    assert len(baseline) == len(effect)

    import scipy.stats
    stat, p = scipy.stats.wilcoxon(baseline, effect, alternative=alternative)

    n = len(baseline)
    w_z = (stat - (n * (n + 1) / 4)) / np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    r = w_z / np.sqrt(n)

    plot_test(
        ax,
        p,
        'W', stat,
        [n],
        desc=f'; r={r:.2f}' if effect_size else None,
        **kwargs,
    )


def mannwhitneyu_test(
        ax,
        baseline, effect,
        alternative='two-sided',
        **kwargs,
):
    import scipy.stats
    u, p = scipy.stats.mannwhitneyu(baseline, effect, alternative=alternative)

    plot_test(
        ax,
        p,
        'U', u,
        [len(baseline), len(effect)],
        **kwargs,

    )
