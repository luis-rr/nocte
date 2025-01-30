"""
Analyse dragon sleep.
"""
import logging

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.signal
import scipy.stats

from nocte import stacks, timeslice
from nocte.stacks import Stack
from nocte.timeslice import S_TO_MS

# nice tutorial: https://raphaelvallat.com/bandpower.html
FREQ_BANDS = pd.DataFrame.from_dict({
    'delta': (0.5, 4, 0),
    'theta': (4, 8, 1),
    'alpha': (8, 12, 2),
    'beta': (12, 30, 3),
    'gamma': (30, 100, 4),
    'spiking': (300, 2_000, 6),
    # 'total': (0, np.inf, 5),
}, orient='index', columns=['freq_min', 'freq_max', 'freq_idx'])
assert FREQ_BANDS['freq_idx'].is_unique


def default_welch_ms(win_len_ms, lowest_freq=0.5):
    """
    Determine a reasonable default value for welch sliding window.

    To determine the Welch sliding window, we take it to include
    at least two full cycles of the lowest frequency of interest.
    For example if the lowest frequency of interest is 0.5 Hz,
    we'll choose: 2 / 0.5 = 4 seconds
    """
    if win_len_ms is None:
        win_len_ms = S_TO_MS * 2 / lowest_freq

    return win_len_ms


def welch(traces: np.array, sampling_rate, win_len_ms=None, db=False, **kwargs):
    """
    If the input signal is in microvolts (uV),
    then the Power Spectral Density  will be in terms of power per frequency,
    expressed as squared microvolts per hertz (uV^2/Hz).
    If db=True, then the result will be in decibels with 1 uV as unit of reference,
    so 10*log10(PSD / 1)
    """
    welch_ms = default_welch_ms(win_len_ms)

    sample_frequences, power_spectral_density = scipy.signal.welch(
        traces,
        axis=0,
        fs=sampling_rate,
        nperseg=int(welch_ms * timeslice.MS_TO_S * sampling_rate),
        **kwargs,
    )

    if db:
        power_spectral_density = 10 * np.log10(power_spectral_density)

    return power_spectral_density, sample_frequences


def band_power(traces: np.array, sampling_rate, bands, welch_ms=None, db=False, add_total=True):
    valid = bands['freq_min'] < (sampling_rate * .5)
    if not valid.all():
        logging.warning(
            f'Bands too high to extract: {bands.index[~valid]} '
            f'Must be below nyquist_freq {sampling_rate * .5})')

    power_spectral_density, sample_frequences = welch(
        traces,
        sampling_rate,
        win_len_ms=welch_ms,
        db=False,
    )

    frequency_resolution = np.diff(sample_frequences)
    assert np.allclose(frequency_resolution, frequency_resolution[0])
    frequency_resolution = frequency_resolution[0]

    # compute the absolute power by approximating the area under the curve
    # using the composite Simpson’s rule
    values = np.empty(
        shape=(len(bands) + (1 if add_total else 0), power_spectral_density.shape[1]),
        dtype=np.float64
    )

    for i, (name, freq_range) in enumerate(bands[['freq_min', 'freq_max']].iterrows()):
        mask = (freq_range['freq_min'] <= sample_frequences) & (sample_frequences <= freq_range['freq_max'])

        assert np.any(mask), \
            f'No frequencies in [{freq_range}]. Extracted: {sample_frequences}'

        # absolute power in uV^2
        sel_power = power_spectral_density[mask, :]
        values[i] = scipy.integrate.simpson(sel_power, dx=frequency_resolution, axis=0)

    # total power
    if add_total:
        values[-1] = scipy.integrate.simpson(power_spectral_density, dx=frequency_resolution, axis=0)

    # build a stack
    # values = values.reshape((-1,) + shape[:-1])

    # prepend freq_band and drop time dimensions
    bands = list(bands.index.values)
    if add_total:
        bands = bands + ['total']

    if db:
        # Decibels are a logarithmic measure of power
        # so we need to first integrate and only after convert to db.
        values = 10 * np.log10(values)

    return values, bands


def extract_power(traces: stacks.Stack, bands=FREQ_BANDS, welch_ms=None, add_total=True):
    """
    Extract the spectral power for each time trace

    :param traces:
    :param bands:
    :param welch_ms: Welch window. By default twice the period of the lowest freq requested.
    :param add_total: wether to total power to the bands index
    :return:
    """
    assert traces.ndim == 2
    assert traces.dims[-1] == 'time'

    power, bands = band_power(
        traces.values.T,
        sampling_rate=traces.estimate_sampling_rate(),
        bands=bands,
        welch_ms=welch_ms,
        add_total=add_total,
    )

    coords = {
        'freq_band': bands,
    }
    # reset_coords will remove non-dimensional coordinates, which we don't want to carry over
    for k, vs in traces.data.reset_coords(drop=True).coords.items():
        if k != 'time':
            coords[k] = vs.values

    return stacks.Stack.from_array(power, coords)


def extract_power_sliding(
        main: Stack,
        sliding_step_ms=1_000.,
        sliding_win_len_ms=20_000.,
        bands=FREQ_BANDS,
        welch_ms=None,
        pbar=None,
        add_total=True,
):
    """
    Extract the spectral power for each time trace using a sliding window.
    Note the windows will overlap and the value generated is assigned to its center.
    This means we cannot cover the beginning and end of the trace.

    :param main:
    :param sliding_step_ms:
        how much the sliding window is shifted on each step, 
        it should be aligned with the sampling rate of the signal
    :param sliding_win_len_ms:
        size of the sliding window

    :param bands: list of names of bands to extract, or full df describing them
    :param add_total:
    :param pbar:
    :param welch_ms:
    :return:
    """
    assert main.get_rel_win().length >= sliding_win_len_ms, \
        f'Data shorter than sliding window ({main.get_rel_win().length} vs {sliding_win_len_ms})'

    if isinstance(bands, list):
        assert all(isinstance(b, str) for b in bands)
        bands = FREQ_BANDS.loc[bands]

    welch_ms = default_welch_ms(welch_ms, bands['freq_min'].min())
    assert sliding_win_len_ms > welch_ms, \
        f'Sliding window ({sliding_win_len_ms} ms) must be bigger than ' \
        f'Welch window ({welch_ms}ms; lowest freq: {bands.freq_min.replace(0, np.nan).min()} Hz)'

    extract_power_kwargs = dict(bands=bands, welch_ms=welch_ms, add_total=add_total)

    sliding_wins = timeslice.Windows.build_sliding_on_stack(
        main,
        length_ms=sliding_win_len_ms,
        step_ms=sliding_step_ms,
    )
    sliding_wins = sliding_wins.wins

    sliding_steps = sliding_wins.index
    if pbar is not None:
        sliding_steps = pbar(sliding_steps, desc='sliding win')

    start = sliding_wins['start']
    stop = sliding_wins['stop']

    all_powers = {}
    for i in sliding_steps:
        section = main.isel(time=slice(start[i], stop[i]))
        ref_ms = sliding_wins.loc[i, 'ref_ms']
        all_powers[ref_ms] = extract_power(section, **extract_power_kwargs)

    all_powers = stacks.stackup(all_powers, 'time')

    # put time first
    all_powers = all_powers.transpose(all_powers.dims[-1], ..., all_powers.dims[0])

    return all_powers


def find_sharp_waves(
        raw_trace: Stack,
        low_pass_hz=30,
        downsample_hz=10,
        width_ms=500.,
        height=2.,
        prominence=1,
        negative=True,
):
    """
    Given a trace in memory, detect sharp-waves in it

    Find SW times using a band pass and find peaks

    :return:
    """
    trace = raw_trace.low_pass(low_pass_hz)
    trace = trace.downsample(downsample_hz)
    trace = trace.zscore()

    peaks = trace.find_peaks(
        width_ms=width_ms,
        height=height,
        prominence=prominence,
        negative=negative,
    )

    return peaks
