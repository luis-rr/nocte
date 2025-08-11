"""
Fast wave Detection in LFP using Template Matching
-------------------------------------------------

This code detects events (e.g. SWR) in local field potential (LFP) recordings
using a fast template-matching approach optimized with NumPy, Numba, and SciPy.

### Approach:
1. **Template Definition**:
   - The user manually selects a few example waves.
   - These examples are averaged to create a representative template.

2. **Sliding Template Correlation**:
   - The template is normalized (z-scored) and slid across the signal.
   - Pearson’s correlation is computed at each time step.
   - A high correlation indicates potential waves detected.

3. **Peak Detection**:
   - The correlation time series is scanned for prominent peaks.
   - Peak properties (height, width, prominence) are extracted.

4. **Filtering**:
   - The process is repeated on a null signal (e.g., vertically inverted) to estimate
     what peak properties look like for "noise".
   - Peaks of the true signal are compared to peaks of the null signal to assign them
     a null CDF value. Picking a p-value to compare to, we filter out peaks that are
     not statistically significantly different from noise.
"""
import logging

import numba
import numba as nb
import numpy as np
import pandas as pd
import scipy.signal

from tqdm.auto import tqdm

from nocte import timeslice
from nocte import traces as tr

from matplotlib import pyplot as plt


@nb.njit(parallel=True)
def extract_cdf_other_nb(this: np.ndarray, null: np.ndarray) -> np.ndarray:
    """
    Estimates the value of the Cumulative Density Function of the distribution
    of the samples in "nul" for each entry of "this".
    Parameters
    ----------
    this
        a 2D numpy array, where rows are samples and columns are features.
    null
        a 2D numpy array, where rows are samples and columns are features.
        The number of features must match that of "this".

    Returns
    -------
        a 1D numpy array, one for each entry of "this" with a value going from 0 to 1
        indicating how likely it is that ALL the features of a given entry of "this"
        are higher than a random sample of "null".

    """
    counts = np.empty(this.shape[0])
    for i in nb.prange(this.shape[0]):

        # noinspection PyTypeChecker
        mask: np.ndarray = (null >= this[i])

        which = np.ones(mask.shape[0], dtype=np.bool_)

        for row in mask.T:
            which = which & row

        count = np.count_nonzero(which)
        counts[i] = count

    return counts / null.shape[0]


@nb.jit(parallel=True)
def _slide_template_nb(signal: np.array, template: np.array):
    """Slide a template calculating pearson's correlation. Fast."""

    template_length = len(template)
    signal_length = len(signal)

    template_std = np.std(template)
    template_mean = np.mean(template)
    template_z = (template - template_mean) / template_std

    scores = np.ones(signal_length - template_length)

    for i in nb.prange(signal_length - template_length):
        section = signal[i:i + template_length]

        section_mean = np.mean(section)
        section_std = np.std(section)

        section_z = (section - section_mean) / section_std

        scores[i] = np.mean(section_z * template_z)

    return scores


def slide_template(signal: pd.Series, template: pd.Series):
    assert len(template) <= len(signal)

    scores = _slide_template_nb(
        signal.values,
        template.values,
    )

    half_template_length = len(template) // 2

    time = signal.index + template.index.min()  # align things to wherever "zero" is in the template time

    scores = pd.Series(
        scores,
        index=time[half_template_length:half_template_length - len(template)],
    )

    return scores


def find_peak_idcs(scores: pd.Series, **kwargs) -> np.ndarray:
    peak_idcs, _ = scipy.signal.find_peaks(scores.values, **kwargs)
    return peak_idcs


def look_up_peak_properties(signal: pd.Series, peak_idcs: np.ndarray) -> pd.DataFrame:
    # Compute prominence (returns left and right base positions)
    prominences, left_bases, right_bases = scipy.signal.peak_prominences(signal.values, peak_idcs, wlen=100)

    # Compute width using prominence bases to ensure correct calculations
    widths, width_heights, left_ips, right_ips = scipy.signal.peak_widths(
        signal.values, peak_idcs,
    )

    peaks = {
        'idx': peak_idcs,
        'height': signal.values[peak_idcs],
        'time': signal.index[peak_idcs],
        'width': widths,
        'width_height': width_heights,
        'prominence': prominences,
        'left_base': left_bases,
        'right_base': right_bases,
        'left_ips': left_ips,
        'right_ips': right_ips,
    }

    return pd.DataFrame(peaks)


def find_closest_peak(peaks, times, col='time'):
    closest_idcs = np.argmin(
        np.abs(peaks[col].values[:, np.newaxis] - times[np.newaxis, :]), axis=0)

    return peaks[col][closest_idcs]


def find_template(
    signal: pd.Series,
    template: pd.Series,
    height_range=(-np.inf, +np.inf),
    width_range=(-np.inf, +np.inf),
    prominence_range=(-np.inf, +np.inf)
):
    scores = slide_template(
        signal,
        template,
    )

    peaks = look_up_peak_properties(
        scores,
        find_peak_idcs(scores),
    )

    peaks['valid'] = (
        peaks['height'].between(*height_range) &
        peaks['width'].between(*width_range) &
        peaks['prominence'].between(*prominence_range)
    )

    # invert the signal to estimate the "noise" distributions
    scores_noise = slide_template(
        signal * -1,
        template,
    )

    peaks_noise = look_up_peak_properties(
        scores_noise,
        find_peak_idcs(scores_noise),
    )

    cdf_cols = ['height', 'width',  'prominence']

    sign = np.array([1, -1, 1])  # invert width because we want narrow peaks

    peaks['null_cdf'] = extract_cdf_other_nb(
        this=peaks[cdf_cols].values * sign,
        null=peaks_noise[cdf_cols].values * sign,
    )

    return peaks


def extract_waves_from_recording(
        loader,
        times,
        cut_out=timeslice.Win(-500, +1500),
        filter_hz=(0.25, 40),
        load_hz=1000,
):
    load_wins = timeslice.Windows.build_around(times, cut_out)

    waves = tr.Traces.load_many(
        loader,
        load_wins,
        load_hz=load_hz,
    )

    waves = waves.filter_pass(filter_hz)

    return waves


def extract_template_from_recording(
        loader,
        times,
        cut_out=timeslice.Win(-500, +1500),
        filter_hz=(0.25, 40),
        load_hz=1000,
):
    template_samples = extract_waves_from_recording(
            loader,
            times,
            cut_out=cut_out,
            filter_hz=filter_hz,
            load_hz=load_hz,
    )

    template = template_samples.mean(axis=1)

    return template


def find_waves_in_recording(
        loader,
        template,
        filter_hz=(0.25, 40),
        load_win=None,
        load_hz=1000,
        channels=None,
        chunk_size=timeslice.ms(hours=1),
        **kwargs,
):
    assert len(loader.channels) == 1

    if load_win is None:
        load_win = loader.win_ms

    breaks = load_win.arange(chunk_size)

    if np.max(breaks) < load_win.stop:
        breaks = np.append(breaks, load_win.stop)

    all_peaks = []

    for t0, t1 in zip(tqdm(breaks[:-1]), breaks[1:]):

        raw = tr.Traces.load_single(
            loader,
            timeslice.Win(t0, t1),
            load_hz=load_hz,
            channels=channels,
        )

        filt = raw.filter_pass(filter_hz)

        signal = filt.get()

        if len(signal) <= len(template):
            assert len(template) <= len(signal)
            logging.warning(
                f'Skipping chunk smaller than template ({len(template)} > {len(signal)}).'
            )
            continue

        peaks = find_template(signal, template, **kwargs)

        peaks['time'] = peaks['time']  + t0

        all_peaks.append(peaks)

    return pd.concat(all_peaks, ignore_index=True)



#########################################################################################################
# Plotting for exploration

def make_scatter_matrix_fig(cols, figsize=(4, 4)):
    f, axs = plt.subplots(
        ncols=len(cols),
        nrows=len(cols),
        sharex='col',
        sharey='row',
        figsize=figsize,
    )

    for j, ax in enumerate(axs[-1, :]):
        ax.set_xlabel(cols[j], fontsize=6)

    for i, ax in enumerate(axs[:, 0]):
        ax.set_ylabel(cols[i], fontsize=6)

    return axs


def plot_scatter_matrix(axs, df, *, max_scatter=10_000, alpha=0.25, s=10, **kwargs):

    xcols = [ax.get_xlabel() for ax in axs[-1, :]]
    ycols = [ax.get_ylabel() for ax in axs[:, 0]]

    for i, ycol in enumerate(xcols):
        for j, xcol in enumerate(ycols):

            ax = axs[i, j]

            if len(df) > max_scatter:
                sdf = df.sample(max_scatter, replace=False)
            else:
                sdf = df

            ax.scatter(
                sdf[xcol],
                sdf[ycol],
                alpha=alpha,
                s=s,
                **kwargs,
            )


def make_hists_fig(cols, figsize=(9, 1.5)):
    f, axs = plt.subplots(ncols=len(cols), figsize=figsize)

    for ax, col in zip(axs, cols):
        ax.set_xlabel(col)

    return axs

def plot_hists_many(axs, df, alpha=0.5, bins=100, density=True, **kwargs):
    for i, ax in enumerate(axs):
        col = ax.get_xlabel()
        ax.hist(
            df[col],
            bins=bins,
            alpha=alpha,
            density=density,
            **kwargs,
        )


@numba.njit
def evaluate_traces_shift_nb(traces: np.array, offsets):
    """
    Align multiple traces by shifting time and baseline

    Parameters
    ----------
    traces
    numpy array of shape <TIME, TRACE>

    offsets
    numpy array of shape <SAMPLED_OFFSET, TRACE, DIM>

    where DIM is either x (time) or y (value)

    Returns
    -------

    """
    assert offsets.shape[1] == traces.shape[1]

    scores = np.empty(offsets.shape[0])

    for i in nb.prange(len(scores)):

        shifted_traces = np.empty(traces.shape, dtype=traces.dtype)

        for j in range(traces.shape[1]):

            offset_x = offsets[i, j, 0]
            offset_y = offsets[i, j, 1]

            trace = traces[:, j]

            shifted_trace = np.roll(trace, offset_x) + offset_y

            shifted_traces[:, j] = shifted_trace

        residual_trace = np.std(shifted_traces, axis=1)

        scores[i] = np.mean(residual_trace)

    return scores
