"""
Code to analyise trains of pulses
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nocte import plot as splot
from nocte import timeslice
from nocte import traces as tr
from nocte.analysis import stim
from nocte.timeslice import Win
from nocte.timeslice import ms, MS_TO_S

PROT_COLORS = pd.Series({
    (80000.0, 1000.0): 'C0',
    (100000.0, 1000.0): 'C1',
    (120000.0, 1000.0): 'C2',
    (140000.0, 1000.0): 'C3',
    (160000.0, 1000.0): 'C4',
    (180000.0, 1000.0): 'C5',
    (200000.0, 1000.0): 'C6',
    (220000.0, 1000.0): 'C8',
    (240000.0, 1000.0): 'C9',
})


def group_pulses(light_wins, max_interval=ms(minutes=5)):
    light_wins = light_wins.sel(cat='on')

    to_next = stim.match_protocol_length(light_wins.interval_to_next())
    to_next.fillna(np.inf, inplace=True)

    to_prev = stim.match_protocol_length(light_wins.interval_to_prev())
    to_prev.fillna(np.inf, inplace=True)

    lengths = stim.match_protocol_length(light_wins.lengths())

    interval = None
    length = None

    k = 0

    group = pd.Series(index=lengths.index, dtype=int)

    for idx in lengths.index:

        if (lengths[idx] != length) or (to_prev[idx] != interval) or interval >= max_interval:
            k += 1

            length = lengths[idx]

            # we use to_next to make sure both are grouped together (if of same length)
            interval = to_next[idx]

        group[idx] = k

    group = group.astype(int)

    return group


def group_pulses_multi(exp_light_wins):
    for exp_name in exp_light_wins.keys():
        light_wins = exp_light_wins[exp_name]
        light_wins['train'] = group_pulses(
            light_wins,
        )


def extract_train_protocol(light_wins):
    light_wins = light_wins.sel(cat='on')

    df = pd.DataFrame({
        'pulse_count': light_wins['train'].value_counts(),
        'pulse_len_precise': light_wins.lengths().groupby(light_wins['train']).median(),
        'start': light_wins.groupby(light_wins['train'])['start'].min(),
        'stop': light_wins.groupby(light_wins['train'])['stop'].max(),
        'interval_precise': light_wins.interval_to_next().groupby(light_wins['train']).median(),
    })

    df.loc[df['pulse_count'] <= 1, 'interval_precise'] = 0

    df['pulse_len'] = stim.match_protocol_length(df['pulse_len_precise'])
    df['interval'] = stim.match_protocol_length(df['interval_precise'])
    df['stop'] = df['stop'] + df['interval']
    df['ref'] = df['start']

    df.index = df.index.astype(int)

    return timeslice.Windows(df)


def _extract_train_protocol_multi(exp_light_wins):
    exp_trains = {}

    for exp_name, light_wins in exp_light_wins.items():
        trains = extract_train_protocol(light_wins)
        # trains = trains.sel_between(pulse_count=(3, np.inf))

        if len(trains) > 0:
            exp_trains[exp_name] = trains

    return exp_trains


def extract_train_protocol_multi(
        exp_light_wins,
        min_pulses=3,
        interval_range=(ms(seconds=0), ms(minutes=10))) -> timeslice.Windows:
    all_trains = _extract_train_protocol_multi(exp_light_wins)

    all_trains = timeslice.Windows.concat(all_trains, local_name='local_train')

    all_trains.wins.rename_axis(index='train_id', inplace=True)
    all_trains.wins.rename(columns={'cycle_idx': 'exp_name'}, inplace=True)

    all_trains = all_trains.sel_between(pulse_count=(min_pulses, np.inf))
    all_trains = all_trains.sel_between(interval=interval_range)

    return all_trains


def get_protocol_desc(interval, pulse_len):
    return f'{pulse_len * MS_TO_S:,g}s every {interval * MS_TO_S:,g}s'


def get_phase_evolution(all_light_wins, valid_trains, exp_phases):
    multiple = {}

    for (interval, pulse_len), trains in valid_trains.iter_groupby(['interval', 'pulse_len']):

        traces = {}

        for train_idx, train_win, props in trains.iter_wins_items():
            light_wins = all_light_wins.sel(exp_name=props['exp_name'], train=props['local_train'], cat='on')

            times = light_wins['start']

            light_phases = exp_phases.sel(exp_name=props['exp_name']).interp(times)

            light_phases = light_phases.add_props(**props)

            ideal_times = np.arange(len(times)) * (interval + props['pulse_len'])

            timing_error = (times - times.min()) - ideal_times
            assert np.all(timing_error < 200), timing_error

            light_phases.traces.index = ideal_times

            base_phase = np.floor(light_phases.traces.loc[0])

            light_phases = light_phases - base_phase

            traces[train_idx] = light_phases

        traces = tr.Traces.concat_dict(traces)

        multiple[interval, pulse_len] = traces

    return multiple


def plot_spread_racorr(all_light_wins, beta_cut, beta_cut_racorrs):
    nrows = len(beta_cut.index)

    f, axs = splot.make_axs_grid_with_marginals(
        figsize=(6, .5 + .75 * nrows),
        nrows=nrows,
        constrained_layout=False,
        ymargin=False,
        size_ratio=5,
    )
    f.tight_layout()
    f.subplots_adjust(hspace=.1, left=.1)
    axs = axs.ravel()

    for i, k in enumerate(beta_cut.index):

        interval = beta_cut.loc[k, 'interval']
        pulse_len = beta_cut.loc[k, 'pulse_len']
        exp_name = beta_cut.loc[k, 'exp_name']
        color = PROT_COLORS[interval, pulse_len]
        racorr = beta_cut_racorrs[k]
        racorr = racorr.dropna()

        ax_dict = axs[i]

        zoom_win = Win(racorr.index.min(), racorr.index.max())

        ax = ax_dict['main']

        splot.plot_racorr(
            ax,
            racorr,
            yscale='seconds',
        )

        ylim = racorr.columns.min(), racorr.columns.max()

        for t in np.concatenate([np.arange(0, ylim[0], -interval), np.arange(interval, ylim[1] + 1, interval)]):
            ax.plot(
                zoom_win,
                [t] * 2,
                linestyle='--',
                zorder=1e5,
                color='k',
                linewidth=.5,
            )
        ax.text(
            0,
            interval,
            f' {interval * MS_TO_S:,g}s',
            va='bottom',
            fontsize=6,
            transform=ax.get_yaxis_transform(),
        )

        splot.add_desc(ax, exp_name, loc='bottom left', loc_pad=0, fontsize=6, bkg_color='none')

        light_wins = all_light_wins.sel(exp_name=exp_name)
        train_win = Win(
            beta_cut.loc[k, 'start'],
            beta_cut.loc[k, 'stop'],
        )

        light_wins = light_wins.sel_between(ref=train_win).sel(cat='on')

        beta_trace = beta_cut.downsample(1_000).get(k).dropna()
        beta_trace = zoom_win.crop_df(beta_trace)

        beta_trace = beta_trace.clip(upper=1.5)
        ax_dict['xmargin'].fill_between(
            beta_trace.index,
            np.zeros(len(beta_trace)),
            beta_trace.values,
            facecolor='k',
            clip_on=False,
        )

        splot.add_yscale_bar(ax_dict['xmargin'])

        for _, win in light_wins.iter_wins():
            win = win.shift(-beta_cut.loc[k, 'ref'])
            win = win.clip(zoom_win)

            for ax in [ax_dict['main'], ax_dict['xmargin']]:
                ax.fill_between(
                    win,
                    [0, 0],
                    [1, 1],
                    alpha=1,
                    edgecolor=color,
                    facecolor=color,
                    linewidth=.5,
                    zorder=1e7,
                    transform=ax.get_xaxis_transform(),
                )

    for axs_dict in axs[:-1]:
        axs_dict['main'].set_xlabel('')
        splot.drop_spine(axs_dict['main'], 'x')

    axs_dict = axs[-1]
    splot.set_time_ticks(axs_dict['main'], tight=True, scale='minutes', major=ms(minutes=5), minor=ms(minutes=1))

    return f


def collect_train_period_lens(exp_rem_wins, analysis_wins, edges='keep'):
    all_periods = []

    for exp_name, trains in tqdm(analysis_wins.iter_groupby('exp_name'), total=analysis_wins['exp_name'].nunique()):
        rem_wins = exp_rem_wins[exp_name]

        rem = rem_wins.sel(cat='rem')
        cut_rem = trains.classify_windows(rem, edges=edges)

        sws = rem_wins.sel(cat='sws')
        cut_sws = trains.classify_windows(sws, edges=edges)

        cut_rem = timeslice.Windows.concat(cut_rem, cycle_name='train_id', local_name='period_id')
        cut_sws = timeslice.Windows.concat(cut_sws, cycle_name='train_id', local_name='period_id')

        cut_rem['exp_name'] = exp_name
        cut_sws['exp_name'] = exp_name

        all_periods.append(cut_rem)
        all_periods.append(cut_sws)

    all_periods = timeslice.Windows.concat_list(all_periods)

    return all_periods
