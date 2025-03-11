"""
Load data from Neuropixels

The acquisition software is SpikeGLX:
 https://github.com/JaneliaSciComp/JRCLUST/wiki/SpikeGLX
 https://billkarsh.github.io/SpikeGLX/

It generates 4 files (any other files in the folder come from sorting):
- *.ap.bin -> raw data (30kHz sampling)
- *.lf.bin -> to look at LFP. Low sampling (2kHz) and low pass filtered (1kHz)
- *.ap.meta *.lf.meta -> metadata

Data is signed 16-bit integer. Row matrix of time (rows) and channel (column):

             chan_1 chan_2  ... chan_n
    time_1:   [[1_1,   2_1, ...,   n_1],
    time_2:    [1_2,   2_2, ...,   n_2],
      ...                   ...,
    time_k:    [1_k,   2_k, ...,   n_k]]


Following code adapted from example provided by supplier, downloaded from:
    https://billkarsh.github.io/SpikeGLX/

"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nocte import timeslice
from nocte.io import common
from nocte.timeslice import MS_TO_S, S_TO_MS

logger = logging.getLogger(__name__)

PICO_TO_MICRO = 1e6


# noinspection PyTypeChecker
def read_meta(meta_path: Path):
    raw = _read_meta_raw(meta_path)

    raw['nSavedChans'] = int(raw['nSavedChans'])

    if 'fileSizeBytes' in raw:
        raw['fileSizeBytes'] = int(raw['fileSizeBytes'])

    else:
        logger.warning(f'Missing "fileSizeBytes" in meta file. Attempting to take value from file system.')
        assert str(meta_path).endswith('.meta')
        bin_path = str(meta_path)[:-len('meta')] + 'bin'
        raw['fileSizeBytes'] = os.stat(bin_path).st_size

    if raw['typeThis'] == 'imec':
        processed = _process_meta_imec(raw)

    elif raw['typeThis'] == 'nidq':
        processed = _process_meta_nidq(raw)

    else:
        raise NotImplementedError(f'Processing of {raw["typeThis"]} is not implemented')

    processed['raw'] = raw
    processed['channel_count'] = raw['nSavedChans']
    processed['sample_count'] = int(raw['fileSizeBytes'] / (2 * processed['channel_count']))

    processed['duration_ms'] = (processed['sample_count'] / processed['sampling_rate']) / MS_TO_S

    assert len(processed['all_channel_ids']) == sum(processed['chan_type_count'].values())

    # A ShankMap locates acq channels geometrically on probe
    # Pattern: "shank:column:row:u" (default single shank, two columns)
    if 'snsShankMap' in raw:
        processed['shank_map'] = pd.DataFrame([
            s.split(':') for s in raw['snsShankMap'][1:-1].split(')(')[1:]],
            columns=['shank', 'column', 'row', 'u']).astype(int)

    else:
        logger.warning(f'Missing "snsShankMap" in meta file.')

    return processed


def _read_meta_raw(meta_path: Path):
    """
    Parse ini file returning a dictionary whose keys are the metadata
    left-hand-side-tags, and values are string versions of the right-hand-side
    metadata values. We remove any leading '~' characters in the tags to match
    the MATLAB version of readMeta.

    The string values are converted to numbers using the "int" and "float"
    fucntions. Note that python 3 has no size limit for integers.
    """

    meta_path = Path(meta_path)
    meta = {}

    with meta_path.open() as f:
        dat_list = f.read().splitlines()

        for m in dat_list:
            cs_list = m.split(sep='=')
            if cs_list[0][0] == '~':
                curr_key = cs_list[0][1:len(cs_list[0])]
            else:
                curr_key = cs_list[0]
            meta.update({curr_key: cs_list[1]})

    return meta


def _process_meta_channelids(meta):
    # Because you can selectively save channels, the ith channel in the file
    # isn't necessarily the ith acquired channel.
    # this converts from ith stored to original index.
    # Note that the SpikeGLX channels are 0 based.
    if meta['snsSaveChanSubset'] == 'all':
        # output = int32, 0 to nSavedChans - 1
        all_channel_ids = np.arange(0, meta['nSavedChans'])
    else:
        # parse the snsSaveChanSubset string
        # split at commas
        ch_str_list = meta['snsSaveChanSubset'].split(sep=',')
        all_channel_ids = np.arange(0, 0)  # creates an empty array of int32

        for sL in ch_str_list:
            curr_list = sL.split(sep=':')
            if len(curr_list) > 1:
                # each set of contiguous channels specified by
                # chan1:chan2 inclusive
                new_chans = np.arange(int(curr_list[0]), int(curr_list[1]) + 1)
            else:
                new_chans = np.arange(int(curr_list[0]), int(curr_list[0]) + 1)

            all_channel_ids = np.append(all_channel_ids, new_chans)

    return all_channel_ids


def _process_meta_imec(meta, prefix='im'):
    processed = {
        'all_channel_ids': _process_meta_channelids(meta),
        'sampling_rate': float(meta[f'{prefix}SampRate']),
    }

    # i2v: multiplicative factor for converting 16-bit file data to voltage.
    # This does not take gain into account.
    # The full conversion with gain is:
    #         dataVolts = dataInt * i2v / gain
    # Note that each channel may have its own gain.
    i2v = float(meta[f'{prefix}AiRangeMax']) / 512

    # counts of each imec channel type that composes the timepoints
    # stored in the binary files.
    chan_type_count_list = meta['snsApLfSy'].split(sep=',')
    chan_type_counts = dict(
        AP=int(chan_type_count_list[0]),
        LF=int(chan_type_count_list[1]),
        SY=int(chan_type_count_list[2]),
    )
    processed['chan_type_count'] = chan_type_counts
    # gain for imec channels.
    # Index into these with the original (acquired) channel IDs.
    ro_table = meta['imroTbl'].split(sep=')')
    # drop header and trailing empty string
    ro_table = ro_table[1:-1]
    ro_table = {int(s[1:].split()[0]): s[1:].split()[1:] for s in ro_table}

    # One entry for each channel plus header entry, plus a final empty entry following the last ')'
    data_channel_count = len(ro_table)
    assert data_channel_count == (chan_type_counts['AP'] + chan_type_counts['LF'])
    ap_gain = np.zeros(data_channel_count)  # default type = float
    lf_gain = np.zeros(data_channel_count)
    if f'{prefix}DatPrb_dock' in meta:
        # NP 2.0; APGain = 80 for all AP
        # return 0 for LFgain (no LF channels)
        ap_gain = ap_gain + 80
    else:
        # 3A, 3B1, 3B2 (NP 1.0)
        for i, (chan_id, entry) in enumerate(ro_table.items()):
            ap_gain[i] = float(entry[2])
            lf_gain[i] = float(entry[3])

    processed['chan_gains'] = {'AP': ap_gain, 'LF': lf_gain}
    # conversion factors scale the data for a particular channel
    # Having accessed a block of raw imec data using makeMemMapRaw, convert
    # values to gain corrected voltages.
    # The conversion is only applied to the saved-channel indicies in chanList.
    # Remember saved-channel indicies are in the range [0:nSavedChans-1].
    # The dimensions of the dataArray remain unchanged.
    # ChanList examples:
    #         [0:AP-1]    all AP channels
    #         [2,6,20]    just these three channels (zero based)
    # Remember that for an lf file, the saved channel indicies (fetched by
    # OriginalChans) will be in the range 384-767 for a standard 3A or 3B probe.
    ap_count = len(processed['chan_gains']['AP'])
    nu_count = ap_count * 2

    conversion_factors = []
    for chan_id in processed['all_channel_ids']:
        # TODO is this right? it looks like a bug (would expect an index, not an id):
        if chan_id < ap_count:
            conv = i2v / processed['chan_gains']['AP'][chan_id]

        elif chan_id < nu_count:
            gain = processed['chan_gains']['LF'][chan_id - ap_count]
            if gain == 0:
                gain = 1
                logger.warning(f'Why is gain 0 ?')
            conv = i2v / gain
        else:
            conv = 1

        # The dataArray contains only the channels in chList
        conversion_factors.append(conv)

    processed['conversion_factor'] = np.array(conversion_factors) * PICO_TO_MICRO

    stand_by_channels = np.array([
        [int(s)] if ':' not in s else np.arange(int(s.split(':')[0]), int(s.split(':')[1]) + 1)
        for s in meta[f'{prefix}Stdby'].split(',') if s
    ])
    if len(stand_by_channels) > 0:
        stand_by_channels = np.concatenate(stand_by_channels)

    processed['standby_channels'] = stand_by_channels

    processed['valid_channels'] = np.array(list(
        set(processed['all_channel_ids']) - set(processed['standby_channels'])))

    # drop SYS channels
    processed['valid_channels'] = processed['valid_channels'][
        processed['valid_channels'] <= (processed['chan_type_count']['AP'] + processed['chan_type_count']['LF'])
        ]

    return processed


def _process_meta_nidq(meta, prefix='ni'):
    processed = {
        'all_channel_ids': _process_meta_channelids(meta),
        'sampling_rate': float(meta[f'{prefix}SampRate']),
        'chan_type_count': {'AP': 1},
        'conversion_factor': np.array([1])
    }

    return processed


def make_memmap_raw(bin_path, channel_count, sample_count, mode='r'):
    # noinspection PyTypeChecker
    return np.memmap(
        bin_path,
        shape=(channel_count, sample_count),
        dtype=np.int16,
        order='F',
        mode=mode,
        offset=0,
    )


def get_probe() -> pd.DataFrame:
    """
    Return a description of the geometry of neuropixels Phase 3B probe (staggered).

    This is a translation of the MATLAB code below (inimec3b_staggered.prb).
    Note matlab indexes starting at 1, which we are calling "channel_id"
    while python starts at 0, which we are calling "channel_idx" (index of the df)

        % Neuropixels Phase 3B probe (staggered)
        % Order of the probe sites in the recording file
        channels = 1:384;

        % Site location in micrometers (x and y)
        geometry = [repmat([43; 11; 59; 27], 96, 1), 20*reshape(repmat((1:192), 2, 1), 384, 1)];

        % Reference sites
        ref_sites = 192;
        channels(ref_sites) = [];
        geometry(ref_sites,:) = [];

        % Recording contact pad size in micrometers. Height x width
        pad = [12 12];

    see probe:

        splot.plot_probe_solid(probe, ylim=(0, 500))

    """
    ref_site_id = 192
    x_spacing = [43, 11, 59, 27]
    y_spacing = 20
    n_electrodes = 384

    probe = pd.DataFrame({
        'x_um': np.tile(x_spacing, int(n_electrodes / len(x_spacing))),
        'y_um': y_spacing * np.repeat(np.arange(1, int(n_electrodes / 2) + 1), 2),
        'channel_id': np.arange(1, n_electrodes + 1),
        'is_ref': False,
    })

    probe.loc[probe['channel_id'] == ref_site_id, 'is_ref'] = True

    probe.rename_axis(index='channel_idx', inplace=True)

    return probe


class DataLoader(common.DataLoader):
    """
    Use like:

        data = DataLoader.load_spikegl(meta_path, bin_path)

        traces = data.load(slice(start, stop), [2, 3, 4])
    """

    def __init__(self, meta, memmap):
        self.meta = meta
        self.memmap = memmap

        self._channels = pd.DataFrame(
            {
                'channel_idx': np.arange(len(self.meta['all_channel_ids'])),
                'is_sys': False
            },
            index=self.meta['all_channel_ids']
        )

        # TODO: we should look up channel index by type. We are after SY0, which can be looked up in channel map.
        # For now just pick up the last channel in the saved file (id 768, index 384)
        if 768 in self.meta['all_channel_ids']:
            self._channels.loc[768, 'is_sys'] = True

        assert len(self.channels) == self.meta['channel_count'], \
            f'Got {len(self.channels)} channels, expected {self.meta.channel_count}'

    @property
    def channels(self) -> pd.DataFrame:
        return self._channels

    @property
    def sample_count(self) -> int:
        return self.meta['sample_count']

    @property
    def sampling_period(self) -> float:
        return self.meta['sampling_period']

    @property
    def sampling_rate_raw(self) -> float:
        return self.meta['sampling_rate']

    @classmethod
    def from_spikegl(cls, meta_path, bin_path, expected_sampling_rate=None):
        meta = read_meta(meta_path)
        meta = pd.Series(meta)
        memmap = make_memmap_raw(bin_path, meta['channel_count'], meta['sample_count'])

        # Sometimes the sampling rate assumed by the spike sorting doesnt match
        # the one stored in the actual raw data file.
        # This allows us to force set it.
        if expected_sampling_rate is not None and not np.isclose(meta['sampling_rate'],
                                                                 expected_sampling_rate, rtol=1e-12):
            original_duration = (meta['sample_count'] / meta['sampling_rate']) / MS_TO_S
            adjusted_duration = (meta['sample_count'] / expected_sampling_rate) / MS_TO_S
            logger.warning(
                f"Adjusting sampling rate from {meta['sampling_rate']} to {expected_sampling_rate} "
                f'which changes duration by {original_duration - adjusted_duration} ms'
            )
            meta['sampling_rate'] = expected_sampling_rate

        meta['sampling_period'] = timeslice.SamplingRate(S_TO_MS / meta['sampling_rate']).adjust_sampling_period()

        bin_file_size = os.path.getsize(bin_path)
        if bin_file_size != meta.raw['fileSizeBytes']:
            logger.warning(f'Expected {meta.raw["fileSizeBytes"]} but found {bin_file_size} in file')

        data = cls(meta, memmap)

        return data

    @classmethod
    def from_interp_data(cls, meta_path, bin_path):
        import json

        with open(meta_path) as f:
            meta = pd.Series(json.load(f))

            if 'sampling_period' not in meta:
                meta['sampling_period'] = timeslice.SamplingRate(meta['sampling_rate']).adjust_sampling_period()

        return cls(
            meta,
            make_memmap_raw(
                bin_path,
                meta['channel_count'],
                meta['sample_count'])
        )

    def load(self, sample_idcs: slice, channels: pd.Index, adjust_gain=True) -> np.ndarray:
        """
        :param channels: list of channel ids
        :param sample_idcs: 
        :param adjust_gain:

        :return: a numpy array of shape <#channels, #samples>
        """
        # TODO check what happens if channels are given in an odd order eg [0, 1] vs [1, 0]
        # noinspection PyTypeChecker
        chan_idcs = self.channels.loc[channels, 'channel_idx']

        print(f'Gonna load: {len(chan_idcs)} x {sample_idcs} samples', flush=True)

        section_raw = self.memmap[chan_idcs, sample_idcs]

        conversion_factor = np.array(self.meta['conversion_factor'])

        if adjust_gain:
            section = section_raw * conversion_factor[chan_idcs, np.newaxis]

            return section

        else:
            # TODO sort this "sys" channel situation
            unique_cf = np.unique(conversion_factor[self.meta['all_channel_ids'][:-1]])

            # TODO store a "unit"!
            if not len(unique_cf):
                logger.warning(f'Gain adjustment not homogeneous accross channels: {unique_cf}')

            return section_raw

    @staticmethod
    def locate_files_spikeglx(probe_path: Path, accept_lf=False) -> (str, str):
        probe_path = Path(probe_path)

        meta_path = list(probe_path.glob('*.ap.meta'))
        if len(meta_path) == 0:
            if accept_lf:
                logger.warning(f'Failed to find ap, trying lf')
                meta_path = list(probe_path.glob('*.lf.meta'))
            else:
                raise FileNotFoundError(f'Expected to find 1 *.ap.meta file. Found {len(meta_path)}.')

        if len(meta_path) != 1:
            raise FileNotFoundError(f'Expected to find 1 meta file. Found {len(meta_path)}.')

        meta_path = meta_path[0]

        bin_path = list(probe_path.glob('*.ap.bin'))
        if len(bin_path) == 0:
            if accept_lf:
                logger.warning(f'Failed to find ap, trying lf')
                bin_path = list(probe_path.glob('*.lf.bin'))
            else:
                raise FileNotFoundError(f'Expected to find 1 *.ap.bin file. Found {len(bin_path)}.')

        if len(bin_path) != 1:
            raise FileNotFoundError(f'Expected to find 1 bin file. Found {len(bin_path)}.')

        bin_path = bin_path[0]

        return meta_path, bin_path


class MultiProbeLoader(common.MultiDataLoader):

    def __init__(self, loaders):
        for probe, loader in loaders.items():
            loader.channels['probe'] = probe

        super().__init__(loaders)

    @staticmethod
    def locate_probes_spikeglx(folder_path: str) -> dict:

        paths = list(Path(folder_path).glob('*_imec*'))
        probe_paths = {}

        for base_path in paths:
            try:
                meta_path, bin_path = DataLoader.locate_files_spikeglx(base_path)
                probe_number = int(str(base_path)[-1])
                probe_paths[probe_number] = (meta_path, bin_path)

            except FileNotFoundError as e:
                logger.error(f'Incomplete folder ({base_path}): {e}')

        return {
            i: probe_paths[i]
            for i in sorted(probe_paths.keys())
        }

    @staticmethod
    def locate_probes_interp(folder_path: str, interp_name='interp') -> dict:

        paths = Path(folder_path).glob(f'{interp_name}/interp_imec_*.meta')

        probe_paths = {}

        for meta_path in paths:
            probe_number = int(str(meta_path)[:-len('.meta')][-1])
            bin_path = Path(str(meta_path).replace('.meta', '.bin'))

            probe_paths[probe_number] = (meta_path, bin_path)

        return {
            i: probe_paths[i]
            for i in sorted(probe_paths.keys())
        }

    @classmethod
    def multiprobe_spikeglx(cls, folder_path):
        paths = cls.locate_probes_spikeglx(folder_path)

        assert len(paths) > 0

        return cls({
            probe: DataLoader.from_spikegl(meta_path, bin_path)
            for probe, (meta_path, bin_path) in paths.items()
        })

    @classmethod
    def multiprobe_interp(cls, folder_path, interp_name='interp'):
        paths = cls.locate_probes_interp(folder_path, interp_name=interp_name)
        if len(paths) == 0:
            raise FileNotFoundError(f'No probe found in {folder_path}')

        return cls({
            probe: DataLoader.from_interp_data(meta_path, bin_path)
            for probe, (meta_path, bin_path) in paths.items()
        })


class DataLoaderBaseline(DataLoader):
    """
    This class will load more data than it is asked for (and thus will be slower), use it to estimate
    the noise, substract it from the data we're asked for, and discard the extra data.
    Alternatively we could pre-process the entire experiment.

    To remove noise from the data, we can substract the median of the signal along two dimensions: channel and time.
    see: https://github.com/cortex-lab/neuropixels/wiki/Recommended_preprocessing

    To estimate correctly each time-point's offset, we take the median across channels.
    To avoid biasing that median by the deep channels (which all look alike in claustrum recordings
    and thus lower the median), we take channels that we estimate to be outside the brain (above channel 250).
    We could compute this offset for the entire experiment, save it and load it on demand.
    That would take ~13GB, so instead we are manually loading the reference every time we load data
    (note that gpfs caching should make this relatively fast)

    To estimate correctly each channel's offset, we need to take the median per channel over a long time window.
    Note that this is in principle not necessary for probes >Phase3A, but I'm seeing a cleaner signal
    so we keep it. Technically we should compute over the entire experiment but quick checks suggest
    that this doesn't change much during an experiment, so we are just taking a window of a few minutes in the
    middle of the experiment. We do this once when initialising the object.
    """

    def __init__(
            self, meta, memmap,
            ref_channels=None,
            ref_time_ms=None,
            ref_duration_mins=10,
            ref_load_hz=None,
    ):
        super().__init__(meta, memmap)
        if ref_channels is None:
            ref_channels = np.arange(250, 384, 1)

        self.ref_channels = ref_channels
        if 768 in self.ref_channels:
            logger.warning(f'Sys channel in ref channels for baseline extraction')

        self.channel_offsets = self._calibrate_channels(
            ref_time_ms,
            ref_duration_mins,
            ref_load_hz,
        )

    def load(self, sample_idcs: slice, channels='all', adjust_gain=True) -> np.ndarray:
        """
        :param channels:
        :param sample_idcs:
        :param adjust_gain:

        :return: a numpy array of shape <#channels, #samples>
        """
        raw = super().load(
            channels=channels,
            sample_idcs=sample_idcs,
            adjust_gain=adjust_gain,
        )

        # noinspection PyTypeChecker
        ref_noise = super().load(
            sample_idcs=sample_idcs,
            channels=self.ref_channels,
            adjust_gain=adjust_gain,
        )

        offset_per_time = np.nanmedian(ref_noise, axis=0)

        denoised = raw - offset_per_time.astype(raw.dtype)

        # noinspection PyTypeChecker
        chan_idcs = self.channels.loc[channels, 'channel_idx']

        offsets = self.channel_offsets[chan_idcs]

        if not adjust_gain:
            offsets = (offsets / self.meta['conversion_factor'][chan_idcs]).astype(denoised.dtype)

        return denoised - offsets[:, np.newaxis]

    def _calibrate_channels(self, ref_ms, duration_min, load_hz):
        """
        Load data for all channels at some point of the experiment and return
        the median per channel (after denoising over time)

        :param ref_ms:
        :param duration_min:
        :return:
        """

        if ref_ms is None:
            ref_ms = self.win_ms.mid

        if load_hz is None:
            load_hz = self.sampling_rate / 30

        load_win_ms = timeslice.Win.build_centered(ref_ms, timeslice.ms(minutes=duration_min))

        load_slice = load_win_ms.to_slice_idx(self.sampling_rate, load_hz)

        # noinspection PyTypeChecker
        data = super().load(
            sample_idcs=load_slice,
            # do this instead of 'all' so we can transparently work with system channels too
            channels=self.channels.index,
            adjust_gain=True,
        )

        # we need to denoise here to be consistent with the use of the offsets later on
        # noinspection PyTypeChecker
        ref_channels_idcs = self.channels.loc[self.ref_channels, 'channel_idx']
        denoised = data - np.nanmedian(data[ref_channels_idcs, :], axis=0)
        channel_offsets = np.nanmedian(denoised, axis=1)

        return channel_offsets
