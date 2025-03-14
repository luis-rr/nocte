"""
Common code to load or save ephys data.
"""
import abc
import logging

import numpy as np
import pandas as pd

from nocte import timeslice
from nocte.timeslice import MS_TO_S, S_TO_MS

logger = logging.getLogger(__name__)


class DataLoader(abc.ABC):

    ##############################################################
    # Abstract interface

    @property
    @abc.abstractmethod
    def sample_count(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def sampling_period(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def channels(self) -> pd.DataFrame:
        """
        A DataFrame with metadata about the channels that this loader
        can load from.
        Index must be unique.
        """
        pass

    @abc.abstractmethod
    def load(self, sample_idcs, channels: pd.Index, adjust_gain=True) -> np.ndarray:
        pass

    ##############################################################
    # Convenience methods

    @staticmethod
    def slice_size(s: slice, total: int):
        """evaluate the expected size of slicing an array"""
        start, stop, step = s.indices(total)
        if step > 0:
            return max(0, (stop - start + step - 1) // step)
        else:
            return max(0, (start - stop - abs(step) + 1) // abs(step))

    @property
    def sampling_rate(self) -> float:
        return S_TO_MS / self.sampling_period

    @property
    def duration_ms(self):
        return (self.sample_count / self.sampling_rate) / MS_TO_S

    @property
    def win_ms(self):
        return timeslice.Win(0, self.duration_ms)

    @property
    def win_idcs(self):
        return timeslice.Win(0, self.sample_count)

    def ms_to_idcs(self, time_ms: np.array):
        """convert absolute timestamps (in ms) into sample indices"""
        time_ms = np.asarray(time_ms) - self.win_ms.start
        return np.round(self.sampling_rate * time_ms * MS_TO_S).astype(int)

    def idcs_to_ms(self, idcs: np.array):
        """convert sample indices into absolute timestamps (in ms)"""
        idcs = np.asarray(idcs)
        time_ms = (idcs / self.sampling_rate) / MS_TO_S
        return time_ms + self.win_ms.start

    def describe(self, quiet=False):
        desc = (
            f' {self.sample_count:,d} samples'
            f' at {self.sampling_rate * MS_TO_S:.2g}kHz'
            f' between {self.win_ms}'
        )

        if quiet:
            return desc
        else:
            print(desc)
            return None


class MultiDataLoader(DataLoader):
    """
    Combine multiple loaders into a single one.

    We are trying to homogenize loading from multiple probes
    independently of recording method:

        - Neuralynx stores one file per channel.
          Each channel has a global unique number: probe 0 contains
          channels 1-32 and probe 1 contains 33-64.
        - Neuropixel stores one file per probe (containing multiple channels).
          Each channel has a *local* unique number: probe 0 contains channels 0-378,
          probe 1 contains channels 0-378.

    To try to make this transparent, we generate a global ID for each channel and
    provide methods to convert between local and global.

    Note that we usually want to refer to channels as pairs (probe, channel),
    not necessarily (file, channel).

    All loaders must be consistent in their sampling rate, and sample count.
    """

    def __init__(self, loaders: dict):
        """
        :param loaders:
            A dictionary of loaders.
            The keys are unique file identifiers (0, 1, 2...) representing probes.
        """
        self.loaders = loaders

        all_channels = {loader_id: loader.channels for loader_id, loader in self.loaders.items()}
        all_channels = pd.concat(all_channels, names=['loader', 'local_channel_id'])
        all_channels = all_channels.sort_index().reset_index()
        all_channels.index.name = 'channel'
        assert all_channels.index.is_unique
        self._channels = all_channels

        all_sampling_rates = pd.Series({
            k: loader.sampling_rate
            for k, loader in self.loaders.items()
        })

        sampling_rate = all_sampling_rates.mean()
        if not all_sampling_rates.nunique() == 1:
            logger.warning(f'Different sampling rate across loaders. Taking mean: {sampling_rate:,.2f}')

        self._sampling_period = timeslice.SamplingRate(sampling_rate).adjust_sampling_period()

        all_sample_counts = pd.Series({
            k: loader.sample_count
            for k, loader in self.loaders.items()
        })
        self._sample_count = all_sample_counts.min()
        if not all_sample_counts.nunique() == 1:
            max_sample_count = all_sample_counts.max()
            logger.error(
                f'Different sample counts across loaders. '
                f'Taking min: {self._sample_count:,g} (loosing {max_sample_count - self._sample_count:,g} samples)'
            )

    def sel_channels(self, which):
        """
        Select a subsection of all channels.
        These will reset the index of channels so
        if which=[32, 64] then the result will have: channels.index == [0, 1]
        :return: a reduced copy of this loader
        """
        new = self.__class__({
            ld: self.loaders[ld]
            for ld in self._channels.loc[which, 'loader'].values
        })
        new._channels = self._channels.loc[which].reset_index(drop=True).rename_axis(index='channel')
        return new

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def sampling_period(self) -> float:
        return self._sampling_period

    @property
    def channels(self) -> pd.DataFrame:
        return self._channels

    def load(self, sample_idcs, channels: pd.Index, adjust_gain=True) -> np.ndarray:

        all_traces = []

        for loader_id, ch_sel in self.channels.loc[channels].groupby('loader'):
            loader = self.loaders[loader_id]

            data = loader.load(
                channels=ch_sel['local_channel_id'].values,
                sample_idcs=sample_idcs,
                adjust_gain=adjust_gain,
            )
            assert data.shape[0] == len(ch_sel)

            all_traces.append(data)

        return np.vstack(all_traces)

    def channel_probes_to_global(self, channels_per_probe):
        """
        Convert a list of channels per probe to global channel ids

        :param channels_per_probe: list of tuples (probe, local channel) like:
            [(0, 224), (1, 32)]

        :return: list of global channel ids
        """
        collect = []
        for probe, ch in channels_per_probe:
            which = (self.channels['probe'] == probe) & (self.channels['local_channel_id'] == ch)
            local_ids = self.channels.index[which]
            if len(local_ids) == 0:
                which = (self.channels['probe'] == probe) & (self.channels['local_channel_id'] == (ch + 1))
                local_ids = self.channels.index[which]

                if len(local_ids) == 0:
                    raise KeyError(f'Failed to locate probe-{probe} channel-{ch}')
                else:
                    logger.warning(f'Channel error by 1')

            assert len(local_ids) == 1
            collect.append(local_ids)

        return np.concatenate(collect)

    def describe(self, quiet=False):

        desc = super().describe(quiet=True)
        desc = f'{len(self.channels):,d} channels ({len(self.loaders):,d} files) {desc}'

        if quiet:
            return desc
        else:
            print(desc)
            return None
