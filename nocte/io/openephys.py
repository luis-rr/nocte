"""
Load data from Open Ephys.

This relies on the open ephys python tools being installed, and will default to that as a loading
mechanism.

"""
import logging

import numpy as np
import pandas as pd
from pathlib import Path

from nocte.io import common
import open_ephys.analysis

logger = logging.getLogger(__name__)


class ContinuousLoader(common.DataLoader):
    """
    Loads data from a single Continuous object from Open Ephys:

    see:
    https://github.com/open-ephys/open-ephys-python-tools/blob/main/src/open_ephys/analysis/formats/BinaryRecording.py
    """

    def __init__(self, stream):
        self.stream = stream

    @property
    def sample_count(self) -> int:
        """The total number of samples in this loader"""
        return self.stream.samples.shape[0]

    @property
    def sampling_period(self) -> float:
        """The inter-sample interval in ms"""
        return 1000. / self.stream.metadata['sample_rate']

    @property
    def channels(self) -> pd.DataFrame:
        """
        A DataFrame with metadata about the channels that this loader can load from.
        Index must be unique and will be used to select channels to load.
        """

        df = pd.DataFrame({
            'name': self.stream.metadata['channel_names'],
        })

        df['local_channel_id'] = df.index

        df.index.name = "channel"

        return df

    def load(self, sample_idcs: slice, channels: pd.Index, adjust_gain=True) -> np.ndarray:
        """
        Load as a numpy array some sample range of some channels.

        :param sample_idcs: a range of samples to load
        :param channels: explicitly what channels to load
        :param adjust_gain: optional, if probe internally stores data as integers
        :return:
        """

        # Note: openephys already adjusts gain for us,
        # so param is ignored but kept for base class compatibility.

        assert isinstance(sample_idcs, slice)

        local_channels = self.channels.loc[channels, 'local_channel_id'].values

        data = self.stream.get_samples(
            sample_idcs.start,
            sample_idcs.stop,
            selected_channels=local_channels,
        )

        return np.asarray(data)

    @classmethod
    def from_session(cls, session_path: str | Path, recording_node_idx=0, recording_idx=0, continuous_idx=0):
        session = open_ephys.analysis.Session(session_path)
        stream = session.recordnodes[recording_node_idx].recordings[recording_idx].continuous[continuous_idx]
        return cls(stream)
