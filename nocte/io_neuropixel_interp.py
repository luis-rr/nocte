"""
Code to perform neuropixel linear interpolation using a stored square signal.
This is necessary when working with multiple probes because, even when callibrated,
the sampling rates differ in a few decimals, which accumulate over several hours of recording.

The recorded square signal is common to everyone and is assumed here to be exactly 30kHz.
The times of this reference are extracted in "extract_onsets" and stored as a csv.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.interpolate
from tqdm.auto import tqdm as pbar

from nocte import timeslice, io_neuropixel

# TODO: we should look up channel index by type. We are after SY0, which can be looked up in channel map.
# For now just pick up the last channel in the saved file (id 768, index 384)
SYS_CHANNEL = 768


def extract_onsets(
        raw: io_neuropixel.DataLoader,
        load_win_ms=None, chunk_length_ms=timeslice.ms(minutes=15)
) -> pd.Series:
    stored_hz = raw.sampling_rate

    if load_win_ms is None:
        load_win_ms = (0, raw.duration_ms)

    chunk_length = timeslice.ms_to_idcs(stored_hz, chunk_length_ms)

    load_win_idcs = (
        timeslice.ms_to_idcs(stored_hz, load_win_ms[0]),
        timeslice.ms_to_idcs(stored_hz, load_win_ms[1]),
    )

    breaks = np.sort(np.unique(np.append(np.arange(*load_win_idcs, chunk_length), load_win_idcs[-1])))

    chunks = zip(breaks[:-1], breaks[1:])

    chunks = pbar(chunks, desc='onsets', total=len(breaks) - 1)

    all_onsets = []
    for i0, i1 in chunks:
        trace = raw.load(channels=pd.Index([SYS_CHANNEL]), sample_idcs=slice(i0, i1))
        trace = trace[0]

        onsets = np.where(np.diff(trace) > 0)[0] + i0

        all_onsets.append(onsets)

    all_onsets = np.sort(np.concatenate(all_onsets))

    return pd.Series(all_onsets).rename('onset')


def extract_onsets_multiprobe(all_raw, load_win_ms):
    all_onset_idcs = pd.DataFrame.from_dict({
        probe: extract_onsets(raw, load_win_ms=load_win_ms)
        for probe, raw in pbar(all_raw.items(), desc='probes')
    })

    all_onset_idcs.sort_index(axis=1, inplace=True)

    return all_onset_idcs


def _make_simple_meta(original, channels, target_hz):
    target_period_ms = timeslice.S_TO_MS / target_hz

    id_to_idx = pd.Series(
        index=original['all_channel_ids'],
        data=np.arange(len(original['all_channel_ids'])),
    )

    channel_idcs = id_to_idx.loc[channels]

    fake_meta = {
        'valid_channels': channels,
        'all_channel_ids': channels,
        'sample_count': int(np.floor(original['duration_ms'] / target_period_ms)),
        'sampling_rate': target_hz,
        'conversion_factor': original['conversion_factor'][channel_idcs],
    }

    fake_meta['channel_count'] = len(fake_meta['all_channel_ids'])
    fake_meta['duration_ms'] = fake_meta['sample_count'] * target_period_ms

    fake_meta = pd.Series(fake_meta)

    return fake_meta


def interpolate_data(meta_path_out, bin_path_out, raw, channels, ref_times, target_hz=30000):
    """
    :param meta_path_out:
    :param bin_path_out:
    :param raw:
    :param channels:
    :param ref_times: a pd.DataFrame indicating the onsets extracted from the system signal.
    Contains 2 columns:
    - idx is the index in the raw data where the onset happened
    - ms is the time, in milliseconds that this sample has in the time frame we want.
    :param target_hz:
    :return:
    """
    target_period_ms = timeslice.S_TO_MS / target_hz

    fake_meta = _make_simple_meta(raw.meta, channels, target_hz)
    fake_meta.to_json(meta_path_out)

    memmap = io_neuropixel.make_memmap_raw(
        bin_path_out, fake_meta['channel_count'], fake_meta['sample_count'], mode='w+')

    chunks = zip(ref_times.index[:-1], ref_times.index[1:])

    chunks = pbar(chunks, total=len(ref_times) - 1, desc='interp')

    for onset0, onset1 in chunks:
        i0 = ref_times.loc[onset0, 'idx']
        i1 = ref_times.loc[onset1, 'idx']

        t0 = ref_times.loc[onset0, 'ms']
        t1 = ref_times.loc[onset1, 'ms']

        # load the edge inclusive to make sure we are within the interpolation range
        true_samples = raw.load(
            channels=channels,
            sample_idcs=slice(i0, i1 + 1),
            adjust_gain=False,
        )
        true_time = np.linspace(t0, t1, true_samples.shape[1])

        lerp = scipy.interpolate.interp1d(true_time, true_samples)
        resample_time = np.arange(t0, t1, target_period_ms)
        resampled_data = lerp(resample_time).astype(true_samples.dtype)

        idcs = timeslice.ms_to_idcs(target_hz, np.array([t0, t1]))
        memmap[:, idcs[0]:idcs[1]] = resampled_data

        memmap.flush()


def multi_npix_from_folder(folder_path, clean=True, quiet=False):
    probe_paths = io_neuropixel.MultiProbeLoader.locate_probes_spikeglx(folder_path)

    all_raw = {}
    for probe_idx, (meta_path, bin_path) in probe_paths.items():

        if clean:
            raw = io_neuropixel.DataLoaderBaseline.from_spikegl(meta_path, bin_path)

        else:
            raw = io_neuropixel.DataLoader.from_spikegl(meta_path, bin_path)

        if not quiet:
            raw.describe()

        all_raw[probe_idx] = raw

    return all_raw


def get_all_onsets(onsets_filename, all_raw, load_win_ms):
    if not onsets_filename.exists():
        all_onset_idcs = extract_onsets_multiprobe(all_raw, load_win_ms)
        all_onset_idcs.to_csv(onsets_filename, index=False, header=True)
    else:
        print(f'Reusing existing onset file:\n{onsets_filename}')

    # note that csv loads column names as string, but we're using numbers for the probes
    all_onset_idcs = pd.read_csv(onsets_filename)
    all_onset_idcs.columns = all_onset_idcs.columns.astype(int)

    # if different probes have different number of onsets
    # the DF will contain nans and values are converted to float,
    # but to use as indices we want them as integers!
    all_onset_idcs = all_onset_idcs.dropna().astype(int)

    return all_onset_idcs


def onset_idcs_to_ms(all_raw, all_onset_idcs):
    all_onset_ms = pd.DataFrame.from_dict({
        probe: timeslice.idcs_to_ms(raw.sampling_rate, all_onset_idcs[probe])
        for probe, raw in all_raw.items()
    })

    return all_onset_ms


def estimate_onset_actual_time(all_raw, all_onset_idcs):
    all_onset_ms = onset_idcs_to_ms(all_raw, all_onset_idcs)

    actual_time = all_onset_ms.mean(axis=1).round()

    # we expect all onsets to be exactly 1 second apart
    # noinspection PyUnresolvedReferences
    if not (actual_time.diff().dropna() == timeslice.S_TO_MS).all():
        actual_time_bad = actual_time

        actual_time = pd.Series(
            np.arange(len(actual_time)) * timeslice.S_TO_MS + actual_time.min(),
            index=actual_time.index,
        )

        desc = ', '.join([
            f'{count} onsets at {val}ms' for val, count in actual_time_bad.dropna().diff().value_counts().items()])

        logging.warning(
            f'Expected onsets to be exactly 1 second apart, but got: {desc}\n'
            f'Adjusting from [{timeslice.strf_ms(actual_time_bad.min())}-{timeslice.strf_ms(actual_time_bad.max())}]'
            f' to [{timeslice.strf_ms(actual_time.min())}-{timeslice.strf_ms(actual_time.max())}]'
        )

    return actual_time


def get_common_onset_times(folder_path, folder_path_out, load_win_ms):
    all_raw_dirt = multi_npix_from_folder(folder_path, clean=False)

    all_onset_idcs = get_all_onsets(
        onsets_filename=folder_path_out / 'onsets.csv',
        all_raw=all_raw_dirt,
        load_win_ms=load_win_ms
    )

    actual_time = estimate_onset_actual_time(all_raw_dirt, all_onset_idcs)

    onsets = pd.concat([all_onset_idcs, actual_time.rename('ms')], axis=1)

    return onsets


def main(
        raw_path: Path,
        channels_per_probe: dict,
        target_hz=30_000,
        output_name='interp',
        load_win_ms=None,
        clean=True,
        overwrite=False,
):
    """
    param channels_per_probe: dict specifying what channels of what probes to interpolate. For example:
        {0: [58], 1: [90], 2: [114]}
    or
        {
            probe: [ch]
            for probe, ch in reg.get_probe_channels(exp_name)
        }

    """

    folder_path_out = raw_path / f'{output_name}/'
    folder_path_out.mkdir(exist_ok=True)
    print(f'Saving results to:\n{folder_path_out}')

    onsets = get_common_onset_times(raw_path, folder_path_out, load_win_ms)

    all_raw = multi_npix_from_folder(raw_path, clean=clean, quiet=True)

    for probe, channels in pbar(channels_per_probe.items(), desc='probes'):

        target_file = folder_path_out / f'interp_imec_{probe}.bin'

        if not target_file.exists() or overwrite:
            raw = all_raw[probe]

            ref_times = onsets[[probe, 'ms']].rename(columns={probe: 'idx'})
            if load_win_ms is not None:
                ref_times = ref_times[ref_times['ms'].between(*load_win_ms)]

            interpolate_data(
                meta_path_out=folder_path_out / f'interp_imec_{probe}.meta',
                bin_path_out=target_file,
                raw=raw,
                channels=list(channels),
                ref_times=ref_times,
                target_hz=target_hz,
            )

        else:
            print(f'Skipping probe {probe} existing file ({target_file}).')
