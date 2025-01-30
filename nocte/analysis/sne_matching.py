"""
Code to match pairs of Sharp Negative Events
"""
import logging

import matplotlib.colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm as pbar

from nocte import plot as splot
from nocte.analysis.sne import SharpNegativeEvents
from nocte.stacks import Stack
from nocte.timeslice import Win, Windows, ms


def _extract_event_pairs_dag(
        xcorr,
        t0s: np.ndarray,
        s0s: np.ndarray,
        t1s: np.ndarray,
        s1s: np.ndarray,
        lag_win,
        min_xcorr: float,
        null_thresh: float,
        show_pbar=True,
) -> np.ndarray:
    """
    Build a matrix representing which events can be matched together,
    where the value is the xcorr sample at each pair of time points

    :param t0s: sorted array of events in channel 0
    :param t1s: sorted array of events in channel 1

        t0s = events.sel_channel(0).reg['ref_time'].sort_values().values
        t1s = events.sel_channel(1).reg['ref_time'].sort_values().values

    :return:
        A matrix of shape (N0, N1) with values containing either an xcorr value
        or np.nan in case the time points are too far away.

    """

    if lag_win is None:
        lag_win = xcorr.get_rel_win('lag')

    value_mat = np.ones((len(t0s), len(t1s))) * np.nan

    i0s = np.arange(len(t0s))
    i1s = np.arange(len(t1s))
    nodes = np.array(np.meshgrid(i0s, i1s)).reshape(2, -1)

    not_noise1 = (s1s[nodes[1]] < null_thresh)
    not_noise0 = (s0s[nodes[0]] < null_thresh)
    valid_noise = not_noise0 | not_noise1

    lags = t1s[nodes[1]] - t0s[nodes[0]]
    valid_lag = (lag_win[0] <= lags) & (lags <= lag_win[1])

    valid = valid_lag & valid_noise

    lags = lags[valid]
    nodes = nodes[:, valid]

    iter_nodes = nodes.T
    if show_pbar:
        iter_nodes = pbar(iter_nodes, desc='get nodes')

    xcorr_values = np.array([
        xcorr.sel(time=t0s[node[0]], lag=lags[i]).data.item()
        for i, node in enumerate(iter_nodes)
    ])

    valid = xcorr_values >= min_xcorr
    xcorr_values = xcorr_values[valid]
    nodes = nodes[:, valid]

    value_mat[nodes[0], nodes[1]] = xcorr_values

    return value_mat


def _find_gradient_dagmat(dagmat, max_ahead, show_pbar=True):
    """
    :param dagmat: this is NOT a connectivity matrix (see extract_event_pairs_dag)

        The DAG that this works on is implicit on the algorithm:
        the nodes are the entries on the matrix, and node (i, j) is
        directly connected to node (k, l) iff (i <= k) & (j <= l).
        Additionally, max_ahead can be specified to limit too far jumps
        (which can make the graph very dense if the ammount of nodes is very big
        and diagonal-like).

    see find_highest_path_dag

    This is similar to Dijkstra's but taking advantage of
    a pre-sorted structure to represent a Directed Acyclic Graph
    between potential pairs of objects.
    """

    # nodes looks like:
    # array([[   2,    7,    8, ..., 5911, 5911, 5912],
    #        [   3,    9,   10, ..., 6405, 6406, 6406]])
    valid_mask = ~np.isnan(dagmat)
    nodes = np.stack(np.where(valid_mask))

    # sort nodes so we process them from the upper-right corner
    nodes_max_coord = -np.max(nodes, axis=0)
    node_sorting = np.argsort(nodes_max_coord)
    nodes = nodes[:, node_sorting]

    cumvalue_mat = np.ones_like(dagmat) * np.nan
    next_node = {}

    iter_nodes = nodes.T
    if show_pbar:
        iter_nodes = pbar(iter_nodes, desc='get nodes')

    for i, j in iter_nodes:
        remaining_values = cumvalue_mat[i + 1:i + 1 + max_ahead, j + 1:j + 1 + max_ahead]

        cumvalue_mat[i, j] = dagmat[i, j]

        if 0 not in remaining_values.shape and np.any(~np.isnan(remaining_values)):
            idx_next = np.nanargmax(remaining_values)
            idx_next = np.unravel_index(idx_next, shape=remaining_values.shape)

            next_node[i, j] = i + 1 + idx_next[0], j + 1 + idx_next[1]

            cumvalue_mat[i, j] += remaining_values[idx_next]

        else:
            next_node[i, j] = None

    root = np.nanargmax(cumvalue_mat)
    root = np.unravel_index(root, shape=cumvalue_mat.shape)

    return root, next_node


def find_highest_path_dag(
        xcorr,
        t0s: np.ndarray,
        s0s: np.ndarray,
        t1s: np.ndarray,
        s1s: np.ndarray,
        null_thresh: float,
        lag_win=None,
        min_xcorr=0,
        max_ahead=None,
        show_pbar=True,
) -> np.array:
    """

    Find the path of maximum value in a Directed Acyclic Graph
    represented as a matrix (see extract_event_pairs_dag).

    Propagate the value of each node from the end to the beginning

    :param max_ahead: number of elements that we are allowed to jump
    :param xcorr:
    :param t0s:
    :param s0s:
    :param t1s:
    :param s1s:
    :param null_thresh:
    :param lag_win:
    :param min_xcorr:
    :param show_pbar:

    :return: a list of nodes (i, j) indicating the path of maximum value


    """
    dagmat = _extract_event_pairs_dag(
        xcorr, t0s, s0s, t1s, s1s, lag_win, min_xcorr,
        null_thresh=null_thresh, show_pbar=show_pbar)

    if max_ahead is None:
        max_ahead = max(*dagmat.shape)

    root, next_node = _find_gradient_dagmat(dagmat, max_ahead, show_pbar=show_pbar)

    path = [root]
    while next_node[path[-1]] is not None:
        path.append(next_node[path[-1]])

    path = np.array(path)

    return path


def _trim_events_outside(xcorr, events):
    valid_win = xcorr.get_rel_win()
    valid_events = events.sel_between(ref_time=valid_win)

    if len(valid_events) != len(events):
        logging.warning(f'{len(events) - len(valid_events)}/{len(events)} events fall outside of xcorr window')
        events = valid_events

    return events


def _calculate_matching(
        events: SharpNegativeEvents,
        xcorr: Stack,
        null_thresh: float,
        ch0=0,
        ch1=1,
        max_ahead=1000,
        show_pbar=True,
) -> np.ndarray:
    e0s = events.sel(channel=ch0).sort_values('ref_time')
    t0s = e0s.reg['ref_time'].values
    s0s = e0s.reg['null_cdf'].values
    assert len(t0s) > 0, f'No SNs in ch{ch0}'

    e1s = events.sel(channel=ch1).sort_values('ref_time')
    t1s = e1s.reg['ref_time'].values
    s1s = e1s.reg['null_cdf'].values
    assert len(t1s) > 0, f'No SNs in ch{ch1}'

    matching = find_highest_path_dag(
        xcorr, t0s, s0s, t1s, s1s,
        null_thresh=null_thresh,
        max_ahead=max_ahead,
        show_pbar=show_pbar,
    )

    matching = np.stack([
        e0s.reg.index[matching[:, 0]],
        e1s.reg.index[matching[:, 1]],
    ])

    return matching


def calculate_matching(
        sns: SharpNegativeEvents,
        full_xcorr: Stack,
        null_thresh=.05,
        chunk_size=ms(minutes=30),
        show_pbar=False,
) -> pd.DataFrame:
    sns = _trim_events_outside(full_xcorr, sns)

    sections = Windows.build_sliding_on_stack(
        full_xcorr,
        chunk_size,
        step_ms=chunk_size,
        ignore_remaining=False,
    )

    missing_times = np.count_nonzero(~sns.reg['ref_time'].isin(full_xcorr.coords['time']))
    if missing_times > 0:
        logging.warning(f'{missing_times}/{len(sns)} events misaligned with xcorr sampling rate. Interpolating xcorr')
        full_xcorr = full_xcorr.interp(np.sort(sns.reg['ref_time'].unique()), pbar=pbar if len(sns) > 1000 else None)

    full_matching = []

    for _, *win in pbar(sections.wins[['start_ms', 'stop_ms']].itertuples(), desc='chunks', total=len(sections)):
        win = Win(*win)

        sns_sel = sns.sel_between(ref_time=win)
        xcorr_sel = full_xcorr.sel_between(time=win)

        if len(sns_sel.reg) > 0:
            full_matching.append(
                _calculate_matching(sns_sel, xcorr_sel, null_thresh=null_thresh, show_pbar=show_pbar)
            )

    full_matching = np.concatenate(full_matching, axis=1)

    matching = pd.DataFrame(full_matching.T, columns=[0, 1])

    assert matching[0].isin(sns.sel(channel=0).reg.index).all()
    assert matching[1].isin(sns.sel(channel=1).reg.index).all()
    assert matching[0].is_unique
    assert matching[1].is_unique

    return matching


def plot_signal(ax, main, linewidth=.5, alpha=.3, orientation='horizontal'):
    for chan in main.coords['channel']:
        x = main.coords['time']
        y = main.sel(channel=chan).values

        if orientation != 'horizontal':
            x, y = y, x

        ax.plot(x, y, color=splot.COLORS[f'ch{chan}'], linewidth=linewidth, alpha=alpha)


def xcorr_to_diagmat(xcorr: Stack) -> Stack:
    """
    convert an xcorr stack (lag x time in ref channel) into a diagonal
    stack of shape (time in ref channel x time in secondary channel).
    Note this is memory intensive since we need to pad the matrix with nans for all
    unknown lags.
    """
    new_mat = np.ones(
        (len(xcorr.coords['time']) * 2 - 1,
         len(xcorr.coords['time']))
    ) * np.nan

    idx = new_mat.shape[0] // 2 - len(xcorr.coords['lag']) // 2
    new_mat[idx:idx + len(xcorr.coords['lag']), :] = xcorr.values

    new_mat = Stack.from_array(
        new_mat, {
            'time1': np.arange(new_mat.shape[0]),
            'time0': np.arange(new_mat.shape[1]),
        })

    shifts = -(np.arange(new_mat.shape[1]) - (new_mat.shape[1] // 2))
    new_mat = new_mat.apply_shift(
        shifts,
        by='time0',
        on='time1',
    )
    new_mat = new_mat.replace_dim('time0', 'time0', xcorr.coords['time'].values)
    new_mat = new_mat.replace_dim('time1', 'time1', xcorr.coords['time'].values[:new_mat.shape[1]])

    return new_mat


def plot_xcorr_diag(main, xcorr, events, zoom_win, show_nodes=True, show_grid=True, y_offset=0):
    xcorr = xcorr.sel_between(time=zoom_win)
    main = main.sel_between(time=zoom_win)
    events = events.sel_between(ref_time=zoom_win)
    diagmat = xcorr_to_diagmat(xcorr)

    f, axs = plt.subplots(
        nrows=2, ncols=1, sharex='col', sharey='row', constrained_layout=True, figsize=(5, 4),
        gridspec_kw=dict(height_ratios=[1, 5])
    )
    #     ax = axs[0, 0]
    #     ax.axis('off')

    axs_dict = {
        'main': axs[1],
        'xmargin': axs[0],
        #         'ymargin': axs[1, 0],
    }

    ax = axs_dict['main']
    vmax = np.nanmax(np.abs(diagmat.values))
    norm = matplotlib.colors.Normalize(-vmax, +vmax)
    ax.imshow(
        diagmat.values,
        norm=norm,
        origin='lower',
        cmap='seismic',
        extent=(
            *diagmat.get_rel_win('time0'),
            *diagmat.get_rel_win('time1'),
        )
    )

    t0 = events.sel(channel=0).reg['ref_time']
    t1 = events.sel(channel=1).reg['ref_time']

    mesh_t0, mesh_t1 = np.meshgrid(t0, t1)

    mesh_t0 = mesh_t0.ravel()
    mesh_t1 = mesh_t1.ravel()
    lag = mesh_t1 - mesh_t0
    valid_lag = (xcorr.get_rel_win('lag')[0] <= lag) & (lag <= xcorr.get_rel_win('lag')[1])

    mesh_t0 = mesh_t0[valid_lag]
    mesh_t1 = mesh_t1[valid_lag]

    ax.plot(
        xcorr.get_rel_win('time'),
        xcorr.get_rel_win('time'),
        color='k',
        linewidth=.5,
    )

    if show_nodes:
        ax.scatter(
            mesh_t0,
            mesh_t1,
            s=50,
            marker='o',
            facecolor='none',
            edgecolor='k',
            linewidth=.25,
        )

    ax.set_xlim(*zoom_win)
    ax.set_ylim(*zoom_win)

    ax = axs_dict['xmargin']
    plot_signal(ax, main.sel(channel=[1]) + y_offset, alpha=1, linewidth=1)
    plot_signal(ax, main.sel(channel=[0]), alpha=1, linewidth=1)

    if show_grid:
        for name in ['main']:
            ax = axs_dict[name]
            ax.vlines(
                events.sel(channel=0).reg['ref_time'], 0, 1,
                color=splot.COLORS['ch0'], transform=ax.get_xaxis_transform(), linewidth=.25, alpha=.25)

        for name in ['main']:
            ax = axs_dict[name]
            ax.hlines(
                events.sel(channel=1).reg['ref_time'], 0, 1,
                color=splot.COLORS['ch1'], transform=ax.get_yaxis_transform(), linewidth=.25, alpha=.25)

    return axs_dict
