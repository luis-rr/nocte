"""
Loading for recordings produced by the Open Ephys GUI.

Open Ephys is a modular electrophysiology acquisition ecosystem used
with Open Ephys hardware, Neuropixels probes, and other compatible recording
hardware. This module delegates file access to the official `open-ephys-python-tools`
package, which provides a common interface across supported Open Ephys recording
formats and streams.

See the Open Ephys documentation and the
(open-ephys-python-tools)[https://github.com/open-ephys/open-ephys-python-tools]
repository for
the authoritative recording-format and Python API descriptions.
"""

from __future__ import annotations

import collections.abc
import importlib
import pathlib
import typing

import numpy as np
import pandas as pd

import nocte.loaders._core
from nocte._core.sampling import SamplingRate

StreamSelector = int | str
StreamSelectors = StreamSelector | collections.abc.Sequence[StreamSelector] | None


class _StreamLoader(nocte.loaders._core.DataLoader):
    """Adapt one Open Ephys continuous stream to the nocte loader contract."""

    def __init__(
        self,
        stream: typing.Any,
        *,
        stream_index: int,
        stream_key: str,
    ) -> None:
        self.stream = stream
        self.stream_index = int(stream_index)
        self.stream_key = str(stream_key)

        metadata = stream.metadata
        channel_count = int(metadata.num_channels)
        if channel_count <= 0:
            raise ValueError('Open Ephys stream must contain at least one channel')

        channel_names = metadata.channel_names
        if channel_names is None:
            channel_names = [None] * channel_count
        elif len(channel_names) != channel_count:
            raise ValueError(
                'Open Ephys channel-name count does not match num_channels'
            )

        bit_volts = np.asarray(metadata.bit_volts, dtype=float)
        if bit_volts.shape != (channel_count,):
            raise ValueError('Open Ephys bit_volts does not match num_channels')
        if not np.isfinite(bit_volts).all() or np.any(bit_volts <= 0):
            raise ValueError('Open Ephys bit_volts must be finite and positive')

        self._bit_volts = bit_volts
        self._sampling = SamplingRate(float(metadata.sample_rate))
        self._signals = pd.DataFrame(
            {
                'stream_index': self.stream_index,
                'stream': self.stream_key,
                'stream_name': str(metadata.stream_name),
                'source_node': int(metadata.source_node_id),
                'source_node_name': str(metadata.source_node_name),
                'channel': np.arange(channel_count, dtype=np.int64),
                'channel_name': channel_names,
                'bit_volts': bit_volts,
                'sampling_hz': self._sampling.rate,
            },
            index=pd.RangeIndex(channel_count, name='signal'),
        )

    @property
    def sample_count(self) -> int:
        sample_numbers = getattr(self.stream, 'sample_numbers', None)
        if sample_numbers is not None:
            return len(sample_numbers)

        return int(self.stream.samples.shape[0])

    @property
    def sampling(self) -> SamplingRate:
        return self._sampling

    @property
    def signals(self) -> pd.DataFrame:
        return self._signals

    def _load_samples(
        self,
        signals: pd.Index,
        start: int,
        stop: int,
        *,
        adjust_gain: bool,
    ) -> np.ndarray:
        local_channels = signals.to_numpy(dtype=np.intp)

        values = np.asarray(
            self.stream.get_samples(
                start_sample_index=start,
                end_sample_index=stop,
                selected_channels=local_channels,
            )
        )

        expected = (stop - start, len(local_channels))
        if values.shape != expected:
            raise RuntimeError(
                f'Open Ephys returned shape {values.shape}, expected {expected}'
            )

        if not adjust_gain:
            values = values / self._bit_volts[local_channels]

        return np.asarray(values).T


class ContinuousLoader(nocte.loaders._core.MultiDataLoader):
    """
    Load continuous signals from one Open Ephys recording.

    One public signal namespace spans one or more selected continuous streams.
    Streams must share compatible sampling rates because nocte aligns them by
    sample index and does not synchronize or resample them.

    Integer channels in :meth:`lookup` are zero-based positions within a
    stream. String channels match Open Ephys channel names exactly.
    """

    def __init__(
        self,
        loaders: collections.abc.Mapping[int, _StreamLoader],
        *,
        sampling_rtol: float = 1e-4,
        sampling_atol: float = 1e-6,
    ) -> None:
        self._stream_loaders = dict(loaders)

        self.recording: typing.Any = None
        self.session: typing.Any = None

        super().__init__(
            self._stream_loaders,
            sampling_rtol=sampling_rtol,
            sampling_atol=sampling_atol,
        )

    def lookup(
        self,
        *,
        stream: StreamSelector,
        channel: int | str,
    ) -> int:
        """Return the unique global signal ID for one stream/channel pair."""
        if isinstance(stream, (int, np.integer)):
            stream_mask = self.signals['stream_index'].eq(int(stream))

        elif isinstance(stream, str):
            stream_mask = self.signals['stream'].eq(stream)

            if not stream_mask.to_numpy(dtype=bool).any():
                stream_mask = self.signals['stream_name'].eq(stream)

        else:
            raise TypeError('stream must be an integer index or string name')

        if isinstance(channel, (int, np.integer)):
            channel_mask = self.signals['channel'].eq(int(channel))

        elif isinstance(channel, str):
            channel_mask = self.signals['channel_name'].eq(channel)

        else:
            raise TypeError('channel must be an integer index or string name')

        matches = self.signals.index[
            stream_mask.to_numpy(dtype=bool) & channel_mask.to_numpy(dtype=bool)
        ]

        if len(matches) == 0:
            raise KeyError(
                f'no Open Ephys signal matches stream={stream!r}, channel={channel!r}'
            )

        if len(matches) > 1:
            raise ValueError(
                f'Open Ephys lookup is ambiguous for stream={stream!r}, '
                f'channel={channel!r}'
            )

        values = matches.to_numpy(dtype=np.intp)
        return int(values[0])

    @classmethod
    def from_recording(
        cls,
        recording: typing.Any,
        *,
        streams: StreamSelectors = 0,
        sampling_rtol: float = 1e-4,
        sampling_atol: float = 1e-6,
    ) -> typing.Self:
        """Build from one ``open_ephys.analysis`` Recording object."""
        available = _continuous_streams(recording)
        selected = _select_streams(available, streams)

        loaders = {
            stream_index: _StreamLoader(
                stream,
                stream_index=stream_index,
                stream_key=stream_key,
            )
            for stream_index, stream_key, stream in selected
        }

        loader = cls(
            loaders,
            sampling_rtol=sampling_rtol,
            sampling_atol=sampling_atol,
        )
        loader.recording = recording
        return loader

    @classmethod
    def from_session(
        cls,
        session_path: str | pathlib.Path,
        *,
        recording_node_idx: int = 0,
        recording_idx: int = 0,
        streams: StreamSelectors = 0,
        sampling_rtol: float = 1e-4,
        sampling_atol: float = 1e-6,
    ) -> typing.Self:
        """
        Build from an Open Ephys session directory.

        ``streams=0`` preserves the old default of loading the first continuous
        stream. Pass a sequence to combine multiple compatible streams, or
        ``streams=None`` to use every continuous stream in the recording.
        """
        session = _load_session(session_path)
        recording = _get_recording(
            session,
            recording_node_idx=recording_node_idx,
            recording_idx=recording_idx,
        )

        loader = cls.from_recording(
            recording,
            streams=streams,
            sampling_rtol=sampling_rtol,
            sampling_atol=sampling_atol,
        )
        loader.session = session
        return loader

    @classmethod
    def find_sources(
        cls,
        session_path: str | pathlib.Path,
    ) -> pd.DataFrame:
        """Inspect continuous streams available under an Open Ephys path."""
        session = _load_session(session_path)
        rows: list[dict[str, object]] = []

        for record_node_idx, recording_idx, recording in _iter_recordings(session):
            for stream_index, stream_key, stream in _continuous_streams(recording):
                metadata = stream.metadata
                sample_numbers = getattr(stream, 'sample_numbers', None)
                if sample_numbers is not None:
                    sample_count = len(sample_numbers)
                else:
                    sample_count = int(stream.samples.shape[0])

                rows.append(
                    {
                        'record_node': record_node_idx,
                        'recording': recording_idx,
                        'stream_index': stream_index,
                        'stream': stream_key,
                        'stream_name': str(metadata.stream_name),
                        'source_node': int(metadata.source_node_id),
                        'source_node_name': str(metadata.source_node_name),
                        'sampling_hz': float(metadata.sample_rate),
                        'channel_count': int(metadata.num_channels),
                        'sample_count': int(sample_count),
                    }
                )

        if not rows:
            raise ValueError(
                f'no Open Ephys continuous streams found in {session_path}'
            )

        result = pd.DataFrame(rows)
        result.index = pd.RangeIndex(len(result), name='source')
        return result


def _load_session(path: str | pathlib.Path) -> typing.Any:
    try:
        analysis = importlib.import_module('open_ephys.analysis')
    except ImportError as error:
        raise ImportError(
            'Open Ephys loading requires the optional `open-ephys-python-tools` package'
        ) from error

    session_class = getattr(analysis, 'Session', None)
    if session_class is None or not callable(session_class):
        raise ImportError('open_ephys.analysis does not expose Session')

    return session_class(str(path))


def _get_recording(
    session: typing.Any,
    *,
    recording_node_idx: int,
    recording_idx: int,
) -> typing.Any:
    if recording_node_idx < 0 or recording_idx < 0:
        raise ValueError('recording indices must be non-negative')

    record_nodes = getattr(session, 'recordnodes', None)
    if record_nodes:
        try:
            recordings = record_nodes[recording_node_idx].recordings
        except IndexError as error:
            raise IndexError(
                f'Open Ephys record node {recording_node_idx} does not exist'
            ) from error
    else:
        if recording_node_idx != 0:
            raise IndexError(
                'this Open Ephys path exposes recordings directly; '
                'recording_node_idx must be 0'
            )
        recordings = getattr(session, 'recordings', None)
        if recordings is None:
            raise ValueError('Open Ephys session exposes no recordings')

    try:
        return recordings[recording_idx]
    except IndexError as error:
        raise IndexError(
            f'Open Ephys recording {recording_idx} does not exist in '
            f'record node {recording_node_idx}'
        ) from error


def _iter_recordings(
    session: typing.Any,
) -> collections.abc.Iterator[tuple[int, int, typing.Any]]:
    record_nodes = getattr(session, 'recordnodes', None)
    if record_nodes:
        for record_node_idx, record_node in enumerate(record_nodes):
            for recording_idx, recording in enumerate(record_node.recordings):
                yield record_node_idx, recording_idx, recording
        return

    recordings = getattr(session, 'recordings', None)
    if recordings is None:
        return

    for recording_idx, recording in enumerate(recordings):
        yield 0, recording_idx, recording


def _continuous_streams(
    recording: typing.Any,
) -> list[tuple[int, str, typing.Any]]:
    continuous = recording.continuous
    if continuous is None or len(continuous) == 0:
        raise ValueError('Open Ephys recording contains no continuous streams')

    stream_names: list[str] = []
    keys = getattr(continuous, 'keys', None)
    if callable(keys):
        raw_keys = typing.cast(collections.abc.Iterable[object], keys())
        stream_names = [key for key in raw_keys if isinstance(key, str)]

    streams: list[tuple[int, str, typing.Any]] = []
    used_names: set[str] = set()

    for stream_index in range(len(continuous)):
        stream = continuous[stream_index]

        if stream_index < len(stream_names):
            stream_key = stream_names[stream_index]
        else:
            stream_key = str(stream.metadata.stream_name)
            if stream_key in used_names:
                stream_key = f'{stream_key}_{int(stream.metadata.source_node_id)}'

        if stream_key in used_names:
            raise ValueError(f'duplicate Open Ephys stream key {stream_key!r}')

        used_names.add(stream_key)
        streams.append((stream_index, stream_key, stream))

    return streams


def _select_streams(
    available: list[tuple[int, str, typing.Any]],
    selectors: StreamSelectors,
) -> list[tuple[int, str, typing.Any]]:
    if selectors is None:
        return available

    if isinstance(selectors, (int, np.integer, str)):
        requested: list[StreamSelector] = [selectors]
    else:
        requested = list(selectors)

    if not requested:
        raise ValueError('at least one Open Ephys stream is required')
    if len(set(requested)) != len(requested):
        raise ValueError('Open Ephys stream selectors must be unique')

    by_index = {item[0]: item for item in available}
    by_name = {item[1]: item for item in available}

    selected: list[tuple[int, str, typing.Any]] = []
    for selector in requested:
        if isinstance(selector, (int, np.integer)):
            key = int(selector)
            if key not in by_index:
                raise KeyError(f'unknown Open Ephys stream index {key}')
            selected.append(by_index[key])
        elif isinstance(selector, str):
            if selector not in by_name:
                raise KeyError(f'unknown Open Ephys stream {selector!r}')
            selected.append(by_name[selector])
        else:
            raise TypeError('stream selectors must be integers or strings')

    return selected
