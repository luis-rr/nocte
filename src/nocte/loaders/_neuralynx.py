"""
Loading for the Neuralynx acquisition ecosystem.

Neuralynx recordings are commonly acquired with Cheetah and Neuralynx hardware
such as Digital Lynx systems, using a variety of compatible electrodes and probes.
Continuous sampled channels are typically stored as NCS files, while events are
stored as NEV files.


See the (Neuralynx Data File Formats)[https://neuralynx.com/_software/NeuralynxDataFileFormats.pdf]
documentation and Cheetah documentation for the authoritative format description.
"""

from __future__ import annotations

import collections.abc
import datetime
import os
import pathlib
import re
import typing
import warnings

import numpy as np
import pandas as pd

import nocte.loaders._core
from nocte._core.sampling import SamplingRate

MICROVOLTS_PER_VOLT = 1_000_000.0
DEFAULT_CHANNELS_PER_PROBE = 64


def _series_value(series: pd.Series, key: str) -> object:
    return typing.cast(object, series.at[key])


def _series_int(series: pd.Series, key: str) -> int:
    value = _series_value(series, key)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f'{key!r} must be integer-like, got {type(value).__name__}')


def _series_float(series: pd.Series, key: str) -> float:
    value = _series_value(series, key)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f'{key!r} must be numeric, got {type(value).__name__}')


def _series_path(series: pd.Series, key: str) -> pathlib.Path:
    value = _series_value(series, key)
    if isinstance(value, pathlib.Path):
        return value
    if isinstance(value, str):
        return pathlib.Path(value)
    raise TypeError(f'{key!r} must be path-like, got {type(value).__name__}')


def _index_first_int(index: pd.Index) -> int:
    values = index.to_numpy(dtype=np.intp)
    if len(values) != 1:
        raise ValueError(f'expected exactly one index value, found {len(values)}')
    return int(values[0])


def _record_int(record: np.void, key: str) -> int:
    value = typing.cast(object, record[key])
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f'{key!r} record field must be an integer scalar')


def _frame_int(frame: pd.DataFrame, row: int, column: str) -> int:
    value = typing.cast(object, frame.at[row, column])
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f'{column!r} must be integer-like, got {type(value).__name__}')


class NeuralynxBaseLoader:
    """Shared Neuralynx header and record helpers."""

    HEADER_LENGTH = 16 * 1024

    @staticmethod
    def load_header(
        path: str | pathlib.Path,
        record_dtype: np.dtype,
    ) -> pd.Series:
        path = pathlib.Path(path)
        with path.open('rb') as fid:
            raw = NeuralynxBaseLoader._read_header(fid)

        header = NeuralynxBaseLoader._parse_header(raw)
        header['path'] = path
        header['record_count'] = NeuralynxBaseLoader._record_count(
            path,
            record_dtype,
        )
        return header

    @staticmethod
    def _read_header(fid: typing.BinaryIO) -> bytes:
        position = fid.tell()
        fid.seek(0)
        raw = fid.read(NeuralynxBaseLoader.HEADER_LENGTH).strip(b'\0')
        fid.seek(position)
        return raw

    @staticmethod
    def _parse_header(raw_header: bytes) -> pd.Series:
        raw = raw_header.decode('iso-8859-1')
        lines = [line.strip() for line in raw.split('\r\n') if line.strip()]
        if not lines:
            raise ValueError('Empty Neuralynx header')

        marker = '######## Neuralynx Data File Header'
        if lines[0] == marker:
            lines = lines[1:]
        else:
            warnings.warn(
                f'Unexpected Neuralynx header start: {lines[0]!r}',
                RuntimeWarning,
                stacklevel=3,
            )

        values: dict[str, object] = {}
        for line in lines:
            split = line.lstrip('-').split(' ', maxsplit=1)
            if len(split) == 2:
                values[split[0]] = split[1]

        def as_bool(value: str) -> object:
            return value.lower() == 'true'

        def as_int(value: str) -> object:
            return int(value)

        def as_float(value: str) -> object:
            return float(value)

        def as_time(value: str) -> object:
            return NeuralynxBaseLoader.parse_time(value)

        casts: dict[str, collections.abc.Callable[[str], object]] = {
            'RecordSize': as_int,
            'SamplingFrequency': as_float,
            'ADMaxValue': as_int,
            'ADBitVolts': as_float,
            'NumADChannels': as_int,
            'ADChannel': as_int,
            'InputRange': as_int,
            'InputInverted': as_bool,
            'DSPLowCutFilterEnabled': as_bool,
            'DspLowCutFrequency': as_float,
            'DSPHighCutFilterEnabled': as_bool,
            'DspHighCutFrequency': as_float,
            'TimeOpened_dt': as_time,
            'TimeClosed_dt': as_time,
        }
        for key, cast in casts.items():
            value = values.get(key)
            if isinstance(value, str):
                values[key] = cast(value)

        for source, target in (
            ('TimeCreated', 'time_created'),
            ('TimeClosed', 'time_closed'),
        ):
            value = values.get(source)
            if not isinstance(value, str):
                continue
            try:
                values[target] = datetime.datetime.strptime(
                    value,
                    '%Y/%m/%d %H:%M:%S',
                ).replace(tzinfo=datetime.UTC)
            except ValueError:
                warnings.warn(
                    f'Could not parse Neuralynx {source}: {value!r}',
                    RuntimeWarning,
                    stacklevel=3,
                )

        return pd.Series(values, dtype=object)

    @staticmethod
    def parse_time(value: object) -> datetime.datetime:
        parts = str(value).split()
        date = [int(part) for part in parts[4].split('/')]
        time = [int(part) for part in parts[-1].replace('.', ':').split(':')]
        return datetime.datetime(
            date[2],
            date[0],
            date[1],
            time[0],
            time[1],
            time[2],
            time[3] * 1000,
            tzinfo=datetime.UTC,
        )

    @staticmethod
    def read_records(
        fid: typing.BinaryIO,
        record_dtype: np.dtype,
        start: int = 0,
        stop: int | None = None,
    ) -> np.ndarray:
        position = fid.tell()
        fid.seek(NeuralynxBaseLoader.HEADER_LENGTH + start * record_dtype.itemsize)
        records = np.fromfile(
            fid,
            record_dtype,
            count=-1 if stop is None else stop - start,
        )
        fid.seek(position)
        return records

    @staticmethod
    def _record_count(path: pathlib.Path, record_dtype: np.dtype) -> int:
        size = os.path.getsize(path) - NeuralynxBaseLoader.HEADER_LENGTH
        if size < 0 or size % record_dtype.itemsize:
            raise ValueError(f'Invalid Neuralynx file size: {path}')
        return size // record_dtype.itemsize

    @staticmethod
    def _channel_from_path(value: object) -> int | None:
        path = str(value).strip().strip('"\'').replace('\\', '/')
        name = path.rsplit('/', maxsplit=1)[-1]
        match = re.fullmatch(r'CSC(\d+)\.ncs', name, flags=re.IGNORECASE)
        return None if match is None else int(match.group(1))

    @staticmethod
    def _channel_from_entity(value: object) -> int | None:
        match = re.fullmatch(r'CSC(\d+)', str(value).strip(), flags=re.IGNORECASE)
        return None if match is None else int(match.group(1))

    @classmethod
    def get_acquisition_channel(cls, header: pd.Series) -> int | None:
        candidates: list[tuple[str, int]] = []

        if 'path' in header:
            value = cls._channel_from_path(_series_value(header, 'path'))
            if value is not None:
                candidates.append(('path', value))

        if 'OriginalFileName' in header:
            value = cls._channel_from_path(_series_value(header, 'OriginalFileName'))
            if value is not None:
                candidates.append(('OriginalFileName', value))

        if 'AcqEntName' in header:
            value = cls._channel_from_entity(_series_value(header, 'AcqEntName'))
            if value is not None:
                candidates.append(('AcqEntName', value))

        if not candidates:
            return None

        if len({value for _, value in candidates}) > 1:
            warnings.warn(
                'Conflicting Neuralynx channel identifiers: '
                + ', '.join(f'{name}={value}' for name, value in candidates)
                + f'; using {candidates[0][0]}={candidates[0][1]}',
                RuntimeWarning,
                stacklevel=3,
            )

        return candidates[0][1]


class NCSLoader(nocte.loaders._core.DataLoader):
    """Load one regular Neuralynx ``.ncs`` file."""

    SAMPLES_PER_RECORD = 512
    RECORD = np.dtype(
        [
            ('TimeStamp', np.uint64),
            ('ChannelNumber', np.uint32),
            ('SampleFreq', np.uint32),
            ('NumValidSamples', np.uint32),
            ('Samples', np.int16, SAMPLES_PER_RECORD),
        ]
    )

    def __init__(self, header: pd.Series) -> None:
        self.header = header.copy()
        self._path = _series_path(self.header, 'path')
        self._record_count = _series_int(self.header, 'record_count')
        self._ad_bit_volts = _series_float(self.header, 'ADBitVolts')

        if (
            'RecordSize' in self.header
            and _series_int(self.header, 'RecordSize') != self.RECORD.itemsize
        ):
            raise ValueError(
                f'Unexpected NCS RecordSize '
                f'{_series_int(self.header, "RecordSize")}; '
                f'expected {self.RECORD.itemsize}'
            )
        if self._record_count <= 0:
            raise ValueError(f'NCS file contains no records: {self._path}')

        first = self._record(0)
        last = self._record(self._record_count - 1)
        first_channel = _record_int(first, 'ChannelNumber')
        last_channel = _record_int(last, 'ChannelNumber')
        if first_channel != last_channel:
            raise ValueError('NCS ChannelNumber changes within the file')

        self._acquisition_channel = self._resolve_channel(first_channel)
        self._sample_count_value = self._sample_count(last)
        self._sampling = SamplingRate(_series_float(self.header, 'SamplingFrequency'))

        self.header['record_channel'] = first_channel
        self.header['acquisition_channel'] = self._acquisition_channel
        self.header['sample_count'] = self._sample_count_value

        self._signals = self.header.to_frame().T
        self._signals['unit'] = 'uV'
        self._signals.index = pd.Index([0], name='signal')

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> typing.Self:
        return cls(NeuralynxBaseLoader.load_header(path, cls.RECORD))

    @property
    def acquisition_channel(self) -> int:
        return self._acquisition_channel

    def _record(self, position: int) -> np.void:
        with self._path.open('rb') as fid:
            records = NeuralynxBaseLoader.read_records(
                fid,
                self.RECORD,
                position,
                position + 1,
            )
        if len(records) != 1:
            raise ValueError(f'Could not read NCS record {position}')
        return typing.cast(np.void, records[0])

    def _resolve_channel(self, record_channel: int) -> int:
        channel = NeuralynxBaseLoader.get_acquisition_channel(self.header)
        if channel is None:
            channel = record_channel + 1
            warnings.warn(
                'Could not identify Neuralynx channel from file/header; '
                f'using record ChannelNumber + 1 ({channel})',
                RuntimeWarning,
                stacklevel=3,
            )
        elif record_channel not in (channel, channel - 1):
            warnings.warn(
                'Neuralynx record/file channel identifiers disagree '
                f'(record={record_channel}, acquisition={channel}); '
                'using the file/header identifier',
                RuntimeWarning,
                stacklevel=3,
            )
        return channel

    def _sample_count(self, last: np.void) -> int:
        valid = _record_int(last, 'NumValidSamples')
        if not 0 <= valid <= self.SAMPLES_PER_RECORD:
            raise ValueError(f'Invalid NumValidSamples: {valid}')
        return (self._record_count - 1) * self.SAMPLES_PER_RECORD + valid

    @property
    def sample_count(self) -> int:
        return self._sample_count_value

    @property
    def sampling(self) -> SamplingRate:
        return self._sampling

    @property
    def signals(self) -> pd.DataFrame:
        return self._signals

    def _read_data(self, start_record: int, stop_record: int) -> np.ndarray:
        with self._path.open('rb') as fid:
            records = NeuralynxBaseLoader.read_records(
                fid,
                self.RECORD,
                start_record,
                stop_record,
            )

        if len(records) != stop_record - start_record:
            raise RuntimeError('Failed to read expected NCS records')
        if not np.allclose(records['SampleFreq'], self.sampling.rate):
            raise ValueError('NCS record sampling rate differs from header')

        valid = records['NumValidSamples'].astype(np.int64)
        positions = np.arange(start_record, stop_record)
        final_record = self._record_count - 1
        irregular = (valid != self.SAMPLES_PER_RECORD) & (positions != final_record)
        if np.any(irregular):
            bad = int(positions[np.flatnonzero(irregular)[0]])
            raise ValueError(
                f'Partially filled intermediate NCS record {bad}; use NCSLoaderUneven'
            )

        data = records['Samples'].reshape(-1)
        if len(records) and positions[-1] == final_record:
            invalid = self.SAMPLES_PER_RECORD - int(valid[-1])
            if invalid:
                data = data[:-invalid]
        return data

    def _load_samples(
        self,
        signals: pd.Index,
        start: int,
        stop: int,
        *,
        adjust_gain: bool,
    ) -> np.ndarray:
        if _index_first_int(signals) != 0:
            raise KeyError('NCSLoader exposes only signal 0')
        if start == stop:
            return np.empty((1, 0), dtype=float if adjust_gain else np.int16)

        rec_start = start // self.SAMPLES_PER_RECORD
        rec_stop = (stop + self.SAMPLES_PER_RECORD - 1) // self.SAMPLES_PER_RECORD
        data = self._read_data(rec_start, rec_stop)

        offset = start - rec_start * self.SAMPLES_PER_RECORD
        data = data[offset : offset + stop - start]
        if len(data) != stop - start:
            raise RuntimeError('Loaded an unexpected number of NCS samples')

        if adjust_gain:
            data = data.astype(np.float64) * (self._ad_bit_volts * MICROVOLTS_PER_VOLT)
        return data.reshape(1, -1)

    def first_timestamp_us(self) -> int:
        return _record_int(self._record(0), 'TimeStamp')


class NCSLoaderUneven(NCSLoader):
    """Load NCS files containing partially filled intermediate records."""

    def __init__(self, header: pd.Series) -> None:
        self._records = self._record_properties(header)
        super().__init__(header)

    @classmethod
    def _record_properties(cls, header: pd.Series) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        count = _series_int(header, 'record_count')
        path = _series_path(header, 'path')

        with path.open('rb') as fid:
            for start in range(0, count, 10_000):
                records = NeuralynxBaseLoader.read_records(
                    fid,
                    cls.RECORD,
                    start,
                    min(start + 10_000, count),
                )
                chunks.append(
                    pd.DataFrame(
                        {
                            'NumValidSamples': records['NumValidSamples'],
                            'SampleFreq': records['SampleFreq'],
                            'TimeStamp': records['TimeStamp'],
                        }
                    )
                )

        if not chunks:
            raise ValueError(f'NCS file contains no records: {path}')

        frame = pd.concat(chunks, ignore_index=True)
        valid = frame['NumValidSamples'].to_numpy(dtype=np.int64)
        frame['stop_sample'] = valid.cumsum()
        frame['first_sample'] = np.concatenate(([0], valid.cumsum()[:-1]))
        return frame

    def _sample_count(self, last: np.void) -> int:
        del last
        return int(self._records['NumValidSamples'].to_numpy(dtype=np.int64).sum())

    def _record_for_sample(self, sample: int) -> int:
        return int(
            np.searchsorted(
                self._records['stop_sample'].to_numpy(dtype=np.int64),
                sample,
                side='right',
            )
        )

    def _read_data(self, start_record: int, stop_record: int) -> np.ndarray:
        with self._path.open('rb') as fid:
            records = NeuralynxBaseLoader.read_records(
                fid,
                self.RECORD,
                start_record,
                stop_record,
            )
        if not np.allclose(records['SampleFreq'], self.sampling.rate):
            raise ValueError('NCS record sampling rate differs from header')

        parts = [
            samples[: int(valid)]
            for valid, samples in zip(
                records['NumValidSamples'],
                records['Samples'],
                strict=True,
            )
        ]
        if not parts:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(parts)

    def _load_samples(
        self,
        signals: pd.Index,
        start: int,
        stop: int,
        *,
        adjust_gain: bool,
    ) -> np.ndarray:
        if _index_first_int(signals) != 0:
            raise KeyError('NCSLoaderUneven exposes only signal 0')
        if start == stop:
            return np.empty((1, 0), dtype=float if adjust_gain else np.int16)

        rec_start = self._record_for_sample(start)
        rec_stop = self._record_for_sample(stop - 1) + 1
        data = self._read_data(rec_start, rec_stop)

        offset = start - _frame_int(self._records, rec_start, 'first_sample')
        data = data[offset : offset + stop - start]
        if len(data) != stop - start:
            raise RuntimeError('Loaded an unexpected number of NCS samples')

        if adjust_gain:
            data = data.astype(np.float64) * (self._ad_bit_volts * MICROVOLTS_PER_VOLT)
        return data.reshape(1, -1)


class NEVLoader:
    """Load a Neuralynx ``.nev`` event file in full."""

    RECORD = np.dtype(
        [
            ('stx', np.int16),
            ('pkt_id', np.int16),
            ('pkt_data_size', np.int16),
            ('TimeStamp', np.uint64),
            ('event_id', np.int16),
            ('ttl', np.int16),
            ('crc', np.int16),
            ('dummy1', np.int16),
            ('dummy2', np.int16),
            ('Extra', np.int32, 8),
            ('EventString', 'S', 128),
        ]
    )

    def __init__(self, header: pd.Series, records: np.ndarray) -> None:
        self.header = header.copy()
        self.records = records

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> typing.Self:
        header = NeuralynxBaseLoader.load_header(path, cls.RECORD)
        with pathlib.Path(path).open('rb') as fid:
            records = NeuralynxBaseLoader.read_records(fid, cls.RECORD)
        return cls(header, records)

    def to_frame(self) -> pd.DataFrame:
        fields = self.records.dtype.fields or {}
        return pd.DataFrame(
            {
                name: self.records[name]
                for name in fields
                if self.records[name].ndim == 1
            }
        )


class MultiNCSLoader(nocte.loaders._core.MultiDataLoader):
    """Recording-level interface over a set of Neuralynx NCS files."""

    def __init__(
        self,
        loaders: collections.abc.Mapping[int, NCSLoader],
        *,
        channels_per_probe: int = DEFAULT_CHANNELS_PER_PROBE,
        sampling_rtol: float = 1e-4,
        sampling_atol: float = 1e-6,
    ) -> None:
        if not isinstance(channels_per_probe, (int, np.integer)):
            raise TypeError('channels_per_probe must be an integer')

        self.channels_per_probe = int(channels_per_probe)

        if self.channels_per_probe <= 0:
            raise ValueError('channels_per_probe must be positive')

        self._ncs_loaders: dict[int, NCSLoader] = dict(loaders)

        super().__init__(
            self._ncs_loaders,
            sampling_rtol=sampling_rtol,
            sampling_atol=sampling_atol,
        )

        channels = self._signals['acquisition_channel'].to_numpy(dtype=np.int64)

        if np.any(channels <= 0) or len(np.unique(channels)) != len(channels):
            raise ValueError(
                'Neuralynx acquisition channels must be unique and positive'
            )

        self._signals['probe'] = (channels - 1) // self.channels_per_probe

        self._signals['channel'] = ((channels - 1) % self.channels_per_probe) + 1

    def lookup(self, *, probe: int, channel: int) -> int:
        """Look up one zero-based probe and one-based probe-local channel."""
        matches = self.signals.index[
            self.signals['probe'].eq(probe) & self.signals['channel'].eq(channel)
        ]
        if len(matches) != 1:
            raise KeyError(
                f'Expected one Neuralynx signal for probe={probe}, '
                f'channel={channel}; found {len(matches)}'
            )
        return _index_first_int(matches)

    @staticmethod
    def find_sources(folder: str | pathlib.Path) -> list[pathlib.Path]:
        folder = pathlib.Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(folder)

        paths = [
            path
            for path in folder.iterdir()
            if path.is_file()
            and re.fullmatch(r'CSC\d+\.ncs', path.name, flags=re.IGNORECASE)
        ]
        return sorted(
            paths,
            key=lambda path: NeuralynxBaseLoader._channel_from_path(path) or -1,
        )

    @classmethod
    def from_paths(
        cls,
        paths: collections.abc.Iterable[str | pathlib.Path],
        *,
        uneven: bool = False,
        channels_per_probe: int = DEFAULT_CHANNELS_PER_PROBE,
        sampling_rtol: float = 1e-4,
        sampling_atol: float = 1e-6,
    ) -> typing.Self:
        loader_class: type[NCSLoader] = NCSLoaderUneven if uneven else NCSLoader
        loaders: dict[int, NCSLoader] = {}

        for path in paths:
            loader = loader_class.from_file(path)
            channel = loader.acquisition_channel
            if channel in loaders:
                raise ValueError(f'Duplicate Neuralynx acquisition channel {channel}')
            loaders[channel] = loader

        if not loaders:
            raise FileNotFoundError('No Neuralynx NCS sources were provided')

        return cls(
            dict(sorted(loaders.items())),
            channels_per_probe=channels_per_probe,
            sampling_rtol=sampling_rtol,
            sampling_atol=sampling_atol,
        )

    @classmethod
    def from_folder(
        cls,
        folder: str | pathlib.Path,
        **kwargs: typing.Any,
    ) -> typing.Self:
        sources = cls.find_sources(folder)
        if not sources:
            raise FileNotFoundError(f'No CSC<n>.ncs files found in {folder}')
        return cls.from_paths(sources, **kwargs)

    def get_joint_header(self) -> pd.DataFrame:
        header = pd.DataFrame.from_dict(
            {source: loader.header for source, loader in self._ncs_loaders.items()},
            orient='index',
        )
        header.index.name = 'source'
        header.columns.name = 'header'
        return header

    def first_timestamps_us(self) -> pd.Series:
        return pd.Series(
            {
                source: loader.first_timestamp_us()
                for source, loader in self._ncs_loaders.items()
            },
            name='first_timestamp_us',
        ).rename_axis('source')
