"""
Code to extract cross correlations between two signals

Convention goes like:

    ch0 = Red = Positive lag leader = Left = lesion = neuronexus 32
    ch1 = Blue = Negative lag leader = Right = control = neuronexus 64

Actual side/lesion/electrode depend on experiment, but usually we match what is above.
In the x-corr we compare ch0 to an offset version of ch1.
If x-corr is high with a positive lag, it means we are comparing ch0 with future ch1, so ch0 (red) leads.

"""

import numpy as np
from numba import njit, prange
from tqdm.auto import tqdm

from nocte import timeslice
from nocte.stacks import Stack
from nocte.timeslice import MS_TO_S


@njit(parallel=True)
def _cross_corr_nb(
        signal0: np.ndarray,
        signal1: np.ndarray,
        idcs: np.ndarray,
        offset: int,
):
    """
    Fast sliding window cross corr.
    See parallelization diagnostics like:
        xc._cross_corr_nb.parallel_diagnostics(level=4)

    :param signal0:
    :param signal1:

    :param idcs:
        shape <N, 2> indicating the pair of indices to cut c0 and c1 to comput xcorr.
        c1 indices will additionally be offset by "offset".

    :param offset:

    :return: array of same length as idcs0
    """

    iter_count = len(idcs)

    corr = np.zeros(iter_count)

    idcs_shifted = idcs + offset

    assert np.min(idcs_shifted[:, 0]) >= 0, 'must zero-pad array or crop windows'
    assert np.max(idcs_shifted[:, 1]) < len(signal1), 'must zero-pad array or crop windows'

    for i in prange(iter_count):
        start0, stop0 = idcs[i]
        start1, stop1 = idcs_shifted[i]

        section0 = signal0[start0:stop0]
        section1 = signal1[start1:stop1]

        # norm = np.std(s0) * np.std(s1)
        # corr[i] = np.mean(s0 * s1) / norm

        corr[i] = np.sum(section0 * section1)

    return corr


@njit(parallel=True)
def _cross_corr_kern_nb(
        signal0: np.ndarray,
        signal1: np.ndarray,
        idcs: np.ndarray,
        kern: np.ndarray,
        offset: int,
):
    """
    Fast sliding window cross corr.
    See parallelization diagnostics like:
        xc._cross_corr_nb.parallel_diagnostics(level=4)

    :param signal0:
    :param signal1:

    :param idcs:
        shape <N, 2> indicating the pair of indices to cut c0 and c1 to comput xcorr.
        c1 indices will additionally be offset by "offset".

    :param offset:

    :return: array of same length as idcs0
    """

    iter_count = len(idcs)

    corr = np.zeros(iter_count)

    idcs_shifted = idcs + offset

    assert np.min(idcs[:, 0]) >= 0, 'must zero-pad array or crop windows'
    assert np.max(idcs[:, 1]) < len(signal0), 'must zero-pad array or crop windows'

    assert np.min(idcs_shifted[:, 0]) >= 0, 'must zero-pad array or crop windows'
    assert np.max(idcs_shifted[:, 1]) < len(signal1), 'must zero-pad array or crop windows'

    for i in prange(iter_count):
        start0, stop0 = idcs[i]
        start1, stop1 = idcs_shifted[i]

        section0 = signal0[start0:stop0]
        section1 = signal1[start1:stop1]

        # norm = np.std(s0) * np.std(s1)
        # corr[i] = np.mean(s0 * s1) / norm

        corr[i] = np.sum(section0 * section1 * kern)

    return corr


def valid_cross_corr(
        signal: Stack,
        lags_ms: np.ndarray,
        win_length_ms=1_000,
        dim='time',
        sliding_step_ms=None,
        show_pbar=True,
        kern=None,
):
    """
    """
    assert signal.ndim == 2
    assert signal.shape[0] == 2, 'Expected only 2 traces in ' + signal.get_coords_names_except(dim)[0]
    lags_ms = np.asarray(lags_ms)

    lag_period_ms = np.diff(lags_ms)
    assert np.allclose(lag_period_ms[0], lag_period_ms)
    lag_period_ms = lag_period_ms[0]

    sampling_period_ms = signal.estimate_sampling_period()
    assert lag_period_ms >= sampling_period_ms, \
        f'Signal sampled every {sampling_period_ms} ms, ' \
        f'but asking for lag resolution of {lag_period_ms} ms'

    sampling_rate = signal.estimate_sampling_rate()

    s0, s1 = signal.values
    lags_idx = np.round(lags_ms * MS_TO_S * sampling_rate).astype(int)

    start_off_ms = - np.min(lags_ms)
    stop_off_ms = +np.max(lags_ms)
    min_valid = start_off_ms + stop_off_ms + win_length_ms

    if signal.get_rel_win().length < min_valid:
        raise ValueError(
            f'Signal must be at least {min_valid}ms. Given {signal.get_rel_win().length}ms')

    sliding_wins = timeslice.Windows.build_sliding_on_stack(
        signal,
        length_ms=win_length_ms,
        step_ms=sliding_step_ms,
        ignore_remaining=True,
        start_off_ms=np.abs(start_off_ms),
        stop_off_ms=-stop_off_ms,
    )
    sliding_wins = sliding_wins.wins
    assert len(sliding_wins) > 0

    valid_wins = (
            (0 <= (sliding_wins['start'] + np.min(lags_idx))) &
            ((np.max(lags_idx) + sliding_wins['stop']) < len(s1))
    )
    assert np.any(valid_wins)

    # print(f'Dropping {np.count_nonzero(~valid_wins)}/{len(valid_wins)} wins where lag becomes invalid')
    sliding_wins = sliding_wins.loc[valid_wins]

    norm = 1  # (np.std(s0) * np.std(s1))

    if show_pbar:
        lags_idx = tqdm(lags_idx, desc='lag')

    idcs = sliding_wins[['start', 'stop']].values.astype(int).copy()
    xcorr = np.empty((len(lags_idx), len(sliding_wins)))

    if kern is not None:
        assert len(kern) == (idcs[0, 1] - idcs[0, 0]), \
            f'Expected kernel of length {(idcs[0, 1] - idcs[0, 0])}. Got {len(kern)}'

    for i, lag_idx in enumerate(lags_idx):
        if kern is None:
            xcorr[i] = _cross_corr_nb(s0, s1, idcs, offset=lag_idx)
        else:
            xcorr[i] = _cross_corr_kern_nb(s0, s1, idcs, offset=lag_idx, kern=kern)

        xcorr[i] = xcorr[i] / norm

    return Stack.from_array(xcorr, {'lag': lags_ms, 'time': sliding_wins['ref_ms']})
