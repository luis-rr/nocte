"""
Process data to extract spectral power, SNs and SWR.
Experiments are long (> 10h) and require processing in chunks or using sliding windows.
"""
import gc
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nocte import timeslice, stacks
from nocte.analysis import sne, sleep, sne_matching
from nocte.analysis import xcorr as xc
from nocte.paths import Registry
from nocte.timeslice import Win, Windows


class ChunkedTraces:
    def __init__(self, filename):
        self.filename = filename

    def store_chunk(self, idx: int, desc: str, trace: pd.Series):
        """"""
        trace.to_hdf(str(self.filename), key=f'{desc}_{idx:03d}')

    def load_all_chunks(self, desc):
        """Returns the list of chunk results of processing the data"""
        with h5py.File(str(self.filename), mode='r') as f:
            entries = sorted([k for k in f.keys() if k.startswith(desc)])

        return [
            pd.read_hdf(self.filename, entry)
            for entry in entries
        ]

    def load_full(self, desc) -> pd.Series:
        """Returns the stitched-together results of processing the data"""
        chunks = self.load_all_chunks(desc)
        assert len(chunks) > 0
        return ChunkedTraces._stitch_results(chunks)

    @staticmethod
    def _stitch_results(chunks) -> pd.Series:
        """
        Take multiple chunk results and stitch them together,
        assuming some potential overlap.
        In case of overlap, the earlier chunk takes precedence.
        """

        if len(chunks) == 1:
            return chunks[0]

        assert len(chunks) > 1

        parts = []

        for i in range(len(chunks) - 1):
            t0 = chunks[i].index.difference(chunks[i + 1].index)
            if len(t0) > 0:
                assert t0.max() < chunks[i + 1].index.min()

            overlap = chunks[i].index.intersection(chunks[i + 1].index)
            t1 = chunks[i + 1].index.difference(chunks[i].index)

            if len(t1) > 0:
                assert t1.min() > chunks[i].index.max()

            if i + 2 < len(chunks):
                t1 = t1.difference(chunks[i + 2].index)

            if i == 0:
                parts.append(chunks[i].loc[t0])

            parts.append((chunks[i].loc[overlap] + chunks[i + 1].loc[overlap]) * .5)
            parts.append(chunks[i + 1].loc[t1])

        # noinspection PyTypeChecker
        stitched: pd.Series = pd.concat(parts)

        # noinspection PyTypeChecker
        final_idx: pd.Index = stitched.index

        assert final_idx.is_unique

        steps = np.unique(np.diff(final_idx))
        assert len(steps) == 1

        return stitched


class ChunkedExperiment:
    """
    Load an experiment in chunks that may overlap,
    compute a signal for each chunk,
    and then stitch them all together.
    """

    def __init__(self, raw, channels, chunk_length, chunk_overlap, load_win=None, load_hz=None, min_chunk_length=None):

        self.raw = raw
        self.channels = channels

        if load_win is None:
            load_win = Win(0, raw.duration_ms)

        assert load_win.length >= 0

        if load_hz is None:
            load_hz = raw.sampling_rate

        load_hz = timeslice.match_load_hz(raw.sampling_rate, load_hz)

        chunk_step_ms = max(chunk_length - chunk_overlap, timeslice.S_TO_MS / load_hz)

        self.stride = timeslice.get_stride(raw.sampling_rate, load_hz)
        load_sampling_period = (self.stride * raw.sampling_period)

        chunk_step_ms = timeslice.adjust_to_sampling_period(chunk_step_ms, load_sampling_period)
        chunk_length = timeslice.adjust_to_sampling_period(chunk_length, load_sampling_period)

        self.wins: Windows = Windows.build_sliding_samples(
            start_ms=load_win.start,
            stop_ms=load_win.stop,
            length_ms=chunk_length,
            sampling_rate=raw.sampling_rate,
            step_ms=chunk_step_ms,
            ignore_remaining=False,
        )

        if min_chunk_length is not None:
            durations_ms: pd.Series = timeslice.S_TO_MS * self.wins.lengths() / raw.sampling_rate

            # noinspection PyTypeChecker
            mask: pd.Series = durations_ms >= min_chunk_length

            if np.any(~mask):
                logging.info(f'Dropping {np.count_nonzero(~mask)}/{len(mask)} chunks smaller than {min_chunk_length}')

            self.wins = self.wins.sel_mask(mask)

        self.load_hz = load_hz

        # assert ((self.wins.lengths() % self.stride).iloc[:-1] == 0).all()

    def iter_chunks(self, pbar_desc='chunk', adjust_gain=True):
        """
        Generator to iterate over chunks
        :param adjust_gain:
        :param pbar_desc:

        :return: a tuple (idx: int, chunk: Stack) of the raw data
            The chunk is a Stack of shape <time, channels>
        """
        win_sections = self.wins.wins[['start', 'stop']].itertuples()
        if pbar_desc is not None:
            win_sections = tqdm(win_sections, total=len(self.wins), desc=pbar_desc)

        for idx, (_, start_idx, stop_idx) in enumerate(win_sections):
            main_raw = self.raw.load(
                channels=self.channels,
                sample_idcs=slice(start_idx, stop_idx, self.stride),
                adjust_gain=adjust_gain,
            )

            t = np.arange(start_idx, stop_idx, self.stride)
            t = t * timeslice.S_TO_MS / self.raw.sampling_rate

            main_raw = stacks.Stack.from_array(
                main_raw,
                {
                    'channel': list(self.channels),
                    'time': t,
                })

            yield idx, main_raw

            gc.collect()

    def get_ref_main(self):
        """
        Load a single window in the middle of the experiment
        """

        duration = self.wins.lengths().iloc[0]

        ref_win_ms = self.raw.idcs_to_ms(self.raw.win_idcs.take_centered(duration).round())

        ref_main = stacks.Stack.load_single_ms(
            self.raw,
            load_win=ref_win_ms,
            load_hz=self.load_hz,
            channels=list(self.channels),
        )

        return ref_main

    def get_ref_zscoring(self):
        """
        To be consistent on the z-score, extract mean and std only once from a middle window
        """
        ref_main = self.get_ref_main()
        ref_main_mean = ref_main.mean('time')
        ref_main_std = ref_main.std('time')

        return ref_main_mean, ref_main_std

    def iter_chunks_zscored(self, *args, **kwargs):
        """
        Generator to iterate over chunks
        :return: a tuple (idx: int, chunk: Stack) of the raw data
            The chunk is a Stack of shape <time, channels>
        """
        ref_main_mean, ref_main_std = self.get_ref_zscoring()

        for idx, main_raw in self.iter_chunks(*args, **kwargs):
            main_zscored = (main_raw - ref_main_mean) / ref_main_std
            yield idx, main_zscored


#################################################################################################################
# x-corr extraction


def extract_xcorr(
        raw,
        ch0, ch1,
        sliding_win,
        low_hz,
        sliding_step=10.,
        lag_range=(-81., +82.),
        mode='speed',
        load_win=None,
        kern=None,
) -> stacks.Stack:
    # we're going heavy here, but I'm tired of trying to do this in chunks
    # and then stitching together the results: because experimental sampling rates often have
    # decimals, it's way too easy to accumulate it and then the stitching isn't *perfect*
    # Instead just load the entire thing in memory

    lags_ms = np.arange(*lag_range, 1.)

    load_hz = 1000

    if load_win is None:
        load_win = raw.win_ms

    main = stacks.Stack.load_single_ms(raw, load_win, load_hz=load_hz, channels=[ch0, ch1])

    main_filtered = main.zscore().low_pass(low_hz)

    if mode == 'speed':
        signal = main_filtered.gradient('time')

    elif mode == 'speed_clipped':
        signal = main_filtered.gradient('time').clip(max=0)

    elif mode == 'acc':
        signal = main_filtered.gradient('time').gradient('time')

    else:
        assert mode == 'raw'
        signal = main_filtered

    xcorr = xc.valid_cross_corr(
        signal, lags_ms,
        win_length_ms=sliding_win,
        sliding_step_ms=sliding_step,
        show_pbar=True,
        kern=kern,
    )

    return xcorr


def process_experiment_xcorr(
        results_path,
        **kwargs,
):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    xcorr = extract_xcorr(**kwargs)

    xcorr.store_hdf(str(results_path), 'xcorr')


def extract_all_xcorr(reg, sliding_win, suffix='', low_hz=40, areas=('CLA', 'BST'), ignore_failures=True, **kwargs):
    import gc

    missing = reg.collect_paths_xcorr(sliding_win, suffix=suffix, missing=True, areas=areas, low_hz=low_hz)

    to_extract = tqdm(missing.itertuples(), desc='experiments', total=len(missing))

    for _, exp_name, results_path, probe0, probe1, ch0, ch1 in to_extract:
        try:
            raw = reg.get_loader(exp_name)

            global_ch = raw.channel_probes_to_global([(probe0, ch0), (probe1, ch1)])

            print(f'Extract:\n{results_path}')
            process_experiment_xcorr(
                results_path,
                raw=raw,
                ch0=global_ch[0],
                ch1=global_ch[1],
                sliding_win=sliding_win,
                low_hz=low_hz,
                **kwargs,
            )

            gc.collect()

        except KeyboardInterrupt:
            raise

        except Exception as e:
            if ignore_failures:
                logging.exception(f'Processing {exp_name} failed: {e}')
            else:
                raise


#################################################################################################################
# power extraction

def extract_sliding(
        raw,
        sliding_win_ms,
        step_ms,
        chunk_length_ms=timeslice.ms(hours=1),
        load_hz=1000,
        channels='all',
        load_win=None,
):
    if load_win is None:
        load_win = raw.win_ms

    load_win = timeslice.Win(*load_win)

    wins_samples = timeslice.Windows.build_sliding_samples(
        load_win.start,
        load_win.stop,
        length_ms=sliding_win_ms,
        sampling_rate=raw.sampling_rate,
        step_ms=step_ms
    )

    wins_ms = wins_samples.sample_to_ms(raw.sampling_rate)

    load_hz = timeslice.match_load_hz(raw.sampling_rate, load_hz)

    chunk_win = Win(0, 0)
    current_chunk = None

    for win_idx in tqdm(wins_ms.wins.index, desc='sliding win'):

        current_win = Win(*wins_ms.wins.loc[win_idx, ['start', 'stop']])

        if not (chunk_win.start <= current_win.start and current_win.stop <= chunk_win.stop):
            chunk_win = Win(current_win.start, current_win.start + chunk_length_ms)

            print('load next chunk:', chunk_win)

            current_chunk = stacks.Stack.load_single_ms(raw, load_win=chunk_win, load_hz=load_hz, channels=channels)

        ref_ms = wins_ms.wins.loc[win_idx, 'ref']
        sel = current_chunk.sel_between(time=current_win)

        yield ref_ms, sel


def extract_sliding_power(
        raw,
        bands=('delta', 'beta', 'spiking'),
        sliding_win_ms=10_000,
        step_ms=1000,
        load_win=None,
        **kwargs,
):
    all_power = {}

    bands = sleep.FREQ_BANDS.loc[list(bands)]

    kwargs['raw'] = raw
    kwargs['sliding_win_ms'] = sliding_win_ms
    kwargs['step_ms'] = step_ms

    for ref_ms, trace in extract_sliding(load_win=load_win, **kwargs):
        all_power[ref_ms] = sleep.extract_power(trace, bands, add_total=False)

    print('merging')
    merged_power = stacks.stackup(all_power, 'time')

    return merged_power


def process_experiment_power(
        results_path,
        raw,
        channel,
        band,
        sliding_win_ms=10_000,
        step_ms=1000,
        load_win=None,
):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    power = extract_sliding_power(
        raw=raw,
        sliding_win_ms=sliding_win_ms,
        step_ms=step_ms,
        channels=[channel],
        bands=[band],
        load_win=load_win,
    )

    power_trace: pd.Series = power.sel(freq_band=band, channel=channel).to_series()

    power_trace.to_hdf(str(results_path), 'power', mode='a')


def extract_all_power(reg, band, sliding_win, sliding_step, areas=('CLA',), ignore_failures=True):
    missing = reg.collect_paths_power(band, sliding_win, sliding_step, missing=True, areas=areas)
    to_extract = tqdm(missing.itertuples(), desc='experiments', total=len(missing))

    for _, exp_name, results_path, probe, local_ch in to_extract:
        try:
            raw = reg.get_loader(exp_name, accept_non_interp=True)

            global_ch = raw.channel_probes_to_global([(probe, local_ch)])

            print(f'Extract:\n{results_path}')
            process_experiment_power(
                results_path,
                raw,
                sliding_win_ms=sliding_win,
                step_ms=sliding_step,
                channel=global_ch[0],
                band=band,
            )

        except KeyboardInterrupt:
            raise

        except Exception as e:
            if ignore_failures:
                logging.exception(f'Processing {exp_name} failed: {e}')
            else:
                raise


#################################################################################################################
# SNE extraction

def process_experiment_sne(
        results_path,
        raw,
        channel,
        load_hz=None,
        chunk_length=timeslice.ms(hours=1),
        load_win=None,
        **kwargs,
):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    chnks = ChunkedExperiment(
        raw=raw,
        load_hz=load_hz,
        load_win=load_win,
        channels=[channel],
        chunk_length=chunk_length,
        chunk_overlap=0,
    )

    all_events_list = []

    print(f'process channel {channel}')

    ref_main = chnks.get_ref_main()
    assert len(ref_main.coords['channel']) == 1
    ref_main = ref_main.isel(channel=0)
    print(f'loaded ref: {ref_main}')

    ref_main_std = ref_main.std('time').values.item()

    sns_null = sne.SharpNegativeEvents.from_double_acceleration(ref_main * -1, **kwargs)
    print(f'extracted {len(sns_null):,g} null SNEs')

    for idx, main in chnks.iter_chunks(pbar_desc='SNEs chunks'):
        assert len(main.coords['channel']) == 1
        main = main.isel(channel=0)

        chunk_events = sne.SharpNegativeEvents.from_double_acceleration(main, **kwargs)
        print(f'extracted {len(chunk_events):,g} SNEs')

        chunk_events['amplitude_zscored'] = chunk_events['amplitude'] / ref_main_std
        chunk_events.reg['null_cdf'] = chunk_events.extract_cdf_other(sns_null)

        all_events_list.append(chunk_events.reg)
        all_events = pd.concat(all_events_list, axis=0, ignore_index=True).rename_axis(index='event_id')
        all_events.to_hdf(results_path, 'sne')


def extract_all_sne(reg, areas=('CLA',), suffix='', ignore_failures=True, load_win=None, missing=True):
    missing = reg.collect_paths_sne(missing=missing, areas=areas, suffix=suffix)
    to_extract = tqdm(missing.itertuples(), desc='experiments', total=len(missing))

    for _, exp_name, results_path, probe, local_ch in to_extract:
        try:
            raw = reg.get_loader(exp_name)

            print(f'Extract:\n{results_path}')
            global_ch = raw.channel_probes_to_global([(probe, local_ch)])[0]

            process_experiment_sne(
                results_path,
                raw,
                load_win=load_win,
                channel=global_ch,
            )
        except KeyboardInterrupt:
            raise

        except Exception as e:
            if ignore_failures:
                logging.exception(f'Processing {exp_name} failed: {e}')
            else:
                raise


def process_experiment_matching(
        results_path,
        reg: Registry, exp_name,
        null_thresh,
        xcorr_sliding_win,
        show_pbar,
):
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Extract {exp_name} matching to: {results_path}')

    xcorr = stacks.Stack.load_hdf(
        reg.get_path_xcorr_area(exp_name, area='CLA', sliding_win=xcorr_sliding_win, suffix='_exp'),
        'xcorr',
    )

    valid_win = xcorr.get_rel_win()

    sns = sne.SharpNegativeEvents(reg.load_all_sne(exp_name, suffix='_cdf'))
    sns = sns.patch_simplified_channels()
    assert sns['channel'].nunique() == 2

    sns = sns.sel_between(ref_time=valid_win)

    # sns['is_valid'] = sns['null_cdf'].between(-np.inf, null_thresh)
    # sns = sns.sel(is_valid=True)

    matching = sne_matching.calculate_matching(
        sns.round(),
        xcorr,
        null_thresh=null_thresh,
        show_pbar=show_pbar,
    )

    matching.to_hdf(str(results_path), 'path')


def extract_all_matchings(
        reg, suffix='', area='',
        null_thresh=.05, xcorr_sliding_win=100,
        show_pbar=False,
        ignore_failures=True,
):
    missing = reg.collect_paths_matching(missing=True, area=area, suffix=suffix)
    to_extract = tqdm(missing.itertuples(), desc='experiments', total=len(missing))

    for _, exp_name, results_path in to_extract:
        try:
            process_experiment_matching(
                results_path, reg, exp_name,
                null_thresh=null_thresh,
                xcorr_sliding_win=xcorr_sliding_win,
                show_pbar=show_pbar,
            )
        except KeyboardInterrupt:
            raise

        except Exception as e:
            if ignore_failures:
                logging.exception(f'Processing {exp_name} failed: {e}')
            else:
                raise
