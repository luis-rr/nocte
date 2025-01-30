"""
Load data from Neuralynx recordings

For formats see
    https://neuralynx.com/_software/NeuralynxDataFileFormats.pdf
    https://support.neuralynx.com/hc/en-us/articles/360030597071-TechTip-Neuralynx-File-Formats
    https://towardsdatascience.com/using-signal-processing-to-extract-neural-events-in-python-964437dc7c0


There are two types of files:
 - nev. Event files. Typically small, one per recording, to synchronize with other methods (like video)
 - ncs. Recording files. Big, one per channel.

classes NCSLoader and NEVLoader simplify access.
class MutiNCSLoader makes it easier to handle multiple NCS files (multiple channels) at once.

NCS data is stored in a series of "records", each with multiple samples.
NCSLoader provides an api that hides these records and lets you index by sample idx.
"""
import datetime
import logging
import os.path
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from nocte import io_common
from nocte import timeslice
from nocte.timeslice import S_TO_MS

VOLT_SCALING = (1, u'V')
MILLIVOLT_SCALING = (1000, u'mV')
MICROVOLT_SCALING = (1000000, u'µV')


class NeuralynxBaseLoader:
    HEADER_LENGTH = 16 * 1024  # (bytes)

    @staticmethod
    def load_header(path: str, record_dtype) -> pd.Series:
        with open(path, 'rb') as fid:
            raw_header = NeuralynxBaseLoader._read_header(fid)
            hdr = NeuralynxBaseLoader._parse_header(raw_header)

        hdr['full_path'] = path
        hdr['record_count'] = NeuralynxBaseLoader._record_count(path, record_dtype)
        hdr['channel_id'] = -1

        return hdr

    @staticmethod
    def _read_header(fid):
        """
        Read the raw header data (16 kb) from the file object fid.
        Restores the position in the file object after reading.

        :return: byte string
        """
        pos = fid.tell()
        fid.seek(0)
        raw_hdr = fid.read(NeuralynxBaseLoader.HEADER_LENGTH).strip(b'\0')
        fid.seek(pos)

        return raw_hdr

    @staticmethod
    def _parse_header(raw_header: bytes) -> pd.Series:
        """
        Parse the header string into a dictionary of name value pairs
        :param raw_header: byte string
        :return: pd.Series
        """

        # Decode the header as iso-8859-1 (the spec says ASCII, but there is at least one case of 0xB5 in some headers)
        raw_hdr = raw_header.decode('iso-8859-1')

        # Neuralynx headers seem to start with a line identifying the file, so
        # let's check for it
        hdr_lines = [line.strip() for line in raw_hdr.split('\r\n') if line != '']

        if hdr_lines[0] != '######## Neuralynx Data File Header':
            logging.warning('Unexpected start to header: ' + hdr_lines[0])
        else:
            hdr_lines = hdr_lines[1:]

        # Assuming "-PARAM_NAME PARAM_VALUE" format
        hdr = {}
        for line in hdr_lines:
            kv = line.lstrip('-').split(' ', maxsplit=1)

            if len(kv) != 2:
                logging.debug(f'Missing value for: "{line}"')
            else:
                hdr[kv[0]] = kv[1]

        def cast_bool(x):
            return x.lower() == 'true'

        castings = {
            'RecordSize': np.int64,
            'SamplingFrequency': np.int64,
            'ADMaxValue': np.int64,
            'ADBitVolts': np.float64,
            'NumADChannels': np.int64,
            'ADChannel': np.int64,
            'InputRange': np.int64,
            'InputInverted': cast_bool,
            'DSPLowCutFilterEnabled': cast_bool,
            'DspLowCutFrequency': np.float64,
            'DspLowCutNumTaps': np.int64,
            'DSPHighCutFilterEnabled': cast_bool,
            'DspHighCutFrequency': np.int64,
            'DspHighCutNumTaps': np.int64,
            'DspFilterDelay_µs': np.int64,
            'TimeOpened_dt': NeuralynxBaseLoader.parse_neuralynx_time_string,
            'TimeClosed_dt': NeuralynxBaseLoader.parse_neuralynx_time_string,
        }

        # Read the parameters, assuming
        keys = list(hdr.keys())

        for k in keys:
            if k in castings:
                hdr[k] = castings[k](hdr[k])

        hdr = pd.Series(hdr)

        # Rename entry for consistency with neuropixel data
        # See also extra properties in NCSLoader.__init__
        if 'SamplingFrequency' in hdr:
            hdr['sampling_rate'] = hdr['SamplingFrequency']
            period = S_TO_MS / hdr['sampling_rate']
            hdr['sampling_period'] = timeslice.adjust_sampling_period(period)

        for old_key, new_key in ('TimeCreated', 'time_created'), ('TimeClosed', 'time_closed'):

            if old_key in hdr:
                try:
                    hdr[new_key] = datetime.datetime.strptime(hdr[old_key], '%Y/%m/%d %H:%M:%S')

                except ValueError as e:
                    logging.warning(f'Failed to parse time {old_key}: "{hdr[old_key]}" ({e})')

        return hdr

    @staticmethod
    def parse_neuralynx_time_string(time_string):
        tmp_date = [int(x) for x in time_string.split()[4].split('/')]
        tmp_time = [int(x) for x in time_string.split()[-1].replace('.', ':').split(':')]
        tmp_microsecond = tmp_time[3] * 1000

        return datetime.datetime(tmp_date[2], tmp_date[0], tmp_date[1],  # Year, month, day
                                 tmp_time[0], tmp_time[1], tmp_time[2],  # Hour, minute, second
                                 tmp_microsecond)

    @staticmethod
    def read_records(fid, record_dtype, start=0, stop=-1) -> np.ndarray:
        """
        Read a number of records (default all) from the file object fid.
        Restores the position of the file object after reading.
        """
        pos = fid.tell()

        flag_abs_position = 0
        fid.seek(NeuralynxBaseLoader.HEADER_LENGTH, flag_abs_position)

        flag_rel_position = 1
        fid.seek(start * record_dtype.itemsize, flag_rel_position)

        rec = np.fromfile(fid, record_dtype, count=stop - start)
        fid.seek(pos)

        return rec

    @staticmethod
    def _record_count(file_path, record_dtype):
        """
        Return the number of records in the file,
        assuming all records are equal in size
        """
        file_size = os.path.getsize(file_path) - NeuralynxBaseLoader.HEADER_LENGTH

        if file_size % record_dtype.itemsize != 0:
            raise ValueError(f'Size of file {file_path} is not multiple of record size.')

        return int(file_size / record_dtype.itemsize)

    @staticmethod
    def get_channel_id(header):
        """Try to extract the channel number from the header. Counting starts with 1."""

        def _channel_id_from_path(path):
            path = str(path).strip().strip('"\'')

            if path.rfind('\\') != -1:
                path = path[path.rfind('\\') + 1:]

            elif path.rfind('/') != -1:
                path = path[path.rfind('/') + 1:]

            if path.startswith('CSC') and path.endswith('.ncs'):
                path = path[len('CSC'):-len('.ncs')]

                if path.isdigit():
                    return int(path)

            return -1

        def _channel_id_from_entity(entity):
            if entity.startswith('CSC'):
                chan = entity[len('CSC'):]

                if chan.isdigit():
                    return int(chan)

            return -1

        chan_number_candidates = np.array([
            _channel_id_from_path(header['OriginalFileName']) if 'OriginalFileName' in header else -1,
            _channel_id_from_entity(header['AcqEntName']) if 'AcqEntName' in header else -1,
            _channel_id_from_path(str(header['full_path'])) if 'full_path' in header else -1,
        ])

        chan_number_candidates = chan_number_candidates[chan_number_candidates != -1]

        if len(chan_number_candidates) == 0:
            logging.error(f'No channel number')
            return None

        chan_number_candidates = np.unique(chan_number_candidates)

        if len(chan_number_candidates) == 1:
            return chan_number_candidates[0]

        else:
            logging.error(f'Multiple possible channel numbers: {list(chan_number_candidates)}')
            return None


class NCSLoader(io_common.DataLoader):
    _SAMPLES_PER_RECORD = 512

    RECORD = np.dtype([
        ('TimeStamp', np.uint64),
        # Cheetah timestamp for this record. This corresponds to
        # the sample time for the first data point in the Samples array.
        # In microseconds.

        ('ChannelNumber', np.uint32),
        # The channel number for this record.
        # NOT the A/D channel number

        ('SampleFreq', np.uint32),
        # The sampling frequency (Hz) for the data stored in the
        # Samples Field in this record

        ('NumValidSamples', np.uint32),
        # Number of values in Samples containing valid data

        ('Samples', np.int16, _SAMPLES_PER_RECORD)
        # Data points for this record. Cheetah
        # currently supports 512 data points per
        # record. At this time, the Samples
        # array is a [512] array.
    ])

    def __init__(self, header):
        self.header = header.copy()
        self.header['sample_count'] = self._get_total_valid_samples()
        self.header['probe'] = self._get_probe_id()

    @classmethod
    def from_file(cls, file_path):
        header = NeuralynxBaseLoader.load_header(file_path, NCSLoader.RECORD)
        header['channel_id'] = NeuralynxBaseLoader.get_channel_id(header)
        if header['channel_id'] is None:
            logging.error(f'Cannot identify channel number for: {file_path}')

        return cls(header)

    @property
    def sample_count(self) -> int:
        return self.header['sample_count']

    @property
    def sampling_rate_raw(self) -> float:
        return self.header['sampling_rate']

    @property
    def sampling_period(self) -> float:
        return self.header['sampling_period']

    @property
    def channels(self) -> pd.DataFrame:
        index = pd.Index([self._get_channel_number() + 1])
        df = self.header.to_frame().T
        return df.set_index(index)

    def _get_probe_id(self) -> int:
        """
        Guess the id of the probe.
        We could parse the header entry for ReferenceChannel e.g. "Headstage 01 Reference 2"
        But it's easier to assume
        :return:
        """
        ch = self._get_channel_number()
        return (ch - 1) // 32

    def _get_total_valid_samples(self) -> int:
        """
        The last record is most likely incomplete, because the recording
        is not necessarily a multiple of 512.

        This opens the last record to check the number of valid samples
        and returns the total number of valid samples in the file
        (assuming all records but the last are full).
        """
        with open(self.header['full_path'], 'rb') as fid:
            records = NeuralynxBaseLoader.read_records(
                fid,
                NCSLoader.RECORD,
                self.header['record_count'] - 1, self.header['record_count']
            )

        valid_in_last = records['NumValidSamples'].item()

        return (self.header['record_count'] - 1) * NCSLoader._SAMPLES_PER_RECORD + valid_in_last

    def _get_channel_number(self) -> int:
        """
        This opens the last record and returns the channel id.
        """
        with open(self.header['full_path'], 'rb') as fid:
            records = NeuralynxBaseLoader.read_records(
                fid,
                NCSLoader.RECORD,
                self.header['record_count'] - 1,
                self.header['record_count'],
            )

        channel = records['ChannelNumber'].item()

        # Note that the stored channel id starts at ZERO, but that the
        # files are saved as starting at ONE,
        # so to be consistent with the file (matlab) convention, we add 1 here
        channel += 1

        return channel

    def _load_records(self, start, stop, adjust_gain=True, verify_timestamps=False) -> np.ndarray:
        """
        Load records by ther index [start, stop)
        Result is in microvolts

        :param start: index of first record
        :param stop: index after last record to load
        :return:
        """
        with open(self.header['full_path'], 'rb') as fid:
            records = NeuralynxBaseLoader.read_records(fid, NCSLoader.RECORD, start, stop)

        assert len(records) > 0
        assert np.all(self.header['SamplingFrequency'] == records['SampleFreq'])

        if len(records) > 1:
            time = (np.mean([start, stop]) / self.sampling_rate) * 1e3

            # ignore last record which may have a clipped count
            all_num_samples = records['NumValidSamples'][:-1]
            num_samples_unique, num_samples_counts = np.unique(all_num_samples, return_counts=True)
            if not np.allclose(num_samples_unique, NCSLoader._SAMPLES_PER_RECORD):
                logging.error(
                    f'Around {time}ms. Expected all records with {NCSLoader._SAMPLES_PER_RECORD} samples. '
                    f'Got ' + ', '.join([
                        f'{c}x{le}'
                        for le, c in zip(num_samples_unique, num_samples_counts)
                    ])
                )

            if verify_timestamps:
                # time stamps come in micro-seconds, one per record
                # make sure that it's the same for all records and that it's consistent with the
                # sampling frequency stated in the header
                all_dts = np.diff(records['TimeStamp']) / all_num_samples
                dts_unique, dts_counts = np.unique(all_dts, return_counts=True)
                expected_dt = (1_000_000 / self.header['SamplingFrequency'])

                if not np.allclose(dts_unique, expected_dt):
                    logging.error(
                        f'Around {time}ms. Expected all samples at {expected_dt} microseconds. '
                        f'Got ' + ', '.join([
                            f'{c}x{dt}' for dt, c in zip(dts_unique, dts_counts)
                        ])
                    )

        # reshape and rescale the data into a 1D array
        data = records['Samples'].ravel()
        assert (len(data) / len(records)) == NCSLoader._SAMPLES_PER_RECORD

        # remove invalid samples at the end of recording
        to_drop = NCSLoader._SAMPLES_PER_RECORD - records['NumValidSamples'][-1]
        to_drop = int(to_drop)
        data = data[:len(data) - to_drop]

        desired_scaling, desired_unit = MICROVOLT_SCALING

        # data comes in "ADC counts"
        # header specifies the conversion factor between the ADC counts and Volts
        # we convert everything to microvolts
        if adjust_gain:
            ad_bit_volts = np.float64(self.header['ADBitVolts'])
            data = data.astype(np.float64) * (ad_bit_volts * desired_scaling)

        return data

    def load(self, sample_idcs, channels=(0,), adjust_gain=True, verify_timestamps=False) -> np.ndarray:
        """
        Load data given a range of sample indices.
        To be consistent with neuropixels (which can load more than one channel at once),
        we take in the fake param "channels" and return a 2-D array
        """
        assert len(channels) == 1
        assert channels[0] == self.channels.index[0]

        assert isinstance(sample_idcs, slice)
        sample_idcs = slice(*sample_idcs.indices(self.header['sample_count']))

        data = self._load_records(
            int(np.floor(sample_idcs.start / NCSLoader._SAMPLES_PER_RECORD)),
            int(np.floor(sample_idcs.stop / NCSLoader._SAMPLES_PER_RECORD)) + 1,
            adjust_gain=adjust_gain,
            verify_timestamps=verify_timestamps,
        )

        start_in_records = sample_idcs.start % NCSLoader._SAMPLES_PER_RECORD

        sample_idcs_in_records = slice(
            start_in_records,
            start_in_records + (sample_idcs.stop - sample_idcs.start),
            sample_idcs.step
        )

        loaded = data[sample_idcs_in_records]

        return loaded.reshape(1, -1)

    def get_first_timestamp(self):
        import datetime

        with open(self.header['full_path'], 'rb') as fid:
            records = NeuralynxBaseLoader.read_records(fid, NCSLoader.RECORD, 0, 1)

        microseconds = records['TimeStamp'].squeeze().item()

        dt = datetime.datetime.fromtimestamp(microseconds // 1000000)
        dt = dt.replace(microsecond=microseconds % 1000000)
        return dt


class NCSLoaderUneven(NCSLoader):
    """
    Loads NCS files where many records are partially full.
    This is slower than NCSLoader, but it's correct.
    """

    def __init__(self, header):
        self._records = NCSLoaderUneven._get_records_props(header)
        super().__init__(header)

    @staticmethod
    def _get_records_props(header, load_step=10_000) -> pd.DataFrame:
        from tqdm.auto import tqdm as pbar

        props = {}
        with open(header['full_path'], 'rb') as fid:
            for r in pbar(np.arange(0, header['record_count'], load_step)):
                records = NeuralynxBaseLoader.read_records(
                    fid,
                    NCSLoader.RECORD,
                    r,
                    r + load_step
                )

                for k in 'TimeStamp', 'NumValidSamples', 'SampleFreq':
                    props.setdefault(k, []).append(records[k])

        props = pd.DataFrame({
            k: np.concatenate(vs)
            for k, vs in props.items()
        })

        props['last_sample_idx'] = props['NumValidSamples'].cumsum()
        props['first_sample_idx'] = np.pad(props['last_sample_idx'].values, (1, 0))[:-1]

        return props

    def _find_records(self, sample_idcs):
        return np.searchsorted(self._records['last_sample_idx'], sample_idcs, side='right')

    def _get_total_valid_samples(self) -> int:
        return self._records['NumValidSamples'].sum()

    def _load_records(self, start, stop, adjust_gain=True, verify_timestamps=False):

        with open(self.header['full_path'], 'rb') as fid:
            records = NeuralynxBaseLoader.read_records(fid, NCSLoader.RECORD, start, stop)

        data = np.concatenate(
            [samples[:num_valid] for num_valid, samples in zip(records['NumValidSamples'], records['Samples'])])

        desired_scaling, desired_unit = MICROVOLT_SCALING

        if adjust_gain:
            ad_bit_volts = np.float64(self.header['ADBitVolts'])
            data = data.astype(np.float64) * (ad_bit_volts * desired_scaling)

        return data

    def load(self, sample_idcs, channels=None, adjust_gain=True, verify_timestamps=False) -> np.ndarray:
        """
        Load data given a range of sample indices.
        To be consistent with neuropixels (which can load more than one channel at once),
        we take in the fake param "channels" and return a 2-D array
        """
        if channels is None:
            channels = (self.channels.index[0],)

        assert len(channels) == 1
        assert channels[0] == self.channels.index[0]

        rec_start, rec_stop = self._find_records([sample_idcs.start, sample_idcs.stop])

        loaded_data = self._load_records(rec_start, rec_stop + 1, verify_timestamps=verify_timestamps)

        off_start = int(sample_idcs.start - self._records.loc[rec_start, 'first_sample_idx'])

        if rec_stop > self._records.index.max():
            off_stop = None

        else:
            off_stop = -int(self._records.loc[rec_stop, 'last_sample_idx'] - sample_idcs.stop)

        data = loaded_data[off_start:off_stop]

        assert (sample_idcs.stop - sample_idcs.start) == len(data), \
            f'Expected {sample_idcs.stop - sample_idcs.start:,d} loaded {len(data):,d}'

        return data[::sample_idcs.step].reshape(1, -1)


class NEVLoader:
    """
    load nev files

    these are typically small and can be loaded in full
    """
    _RECORD = np.dtype([
        ('stx', np.int16),
        # Reserved

        ('pkt_id', np.int16),
        # ID for the originating system of this packet

        ('pkt_data_size', np.int16),
        # This value should always be two (2)

        ('TimeStamp', np.uint64),
        # Cheetah timestamp for this record. This value is in microseconds.

        ('event_id', np.int16),
        # ID value for this event

        ('ttl', np.int16),
        # Decimal TTL value read from the TTL input port

        ('crc', np.int16),
        # Record CRC check from Cheetah. Not used in consumer  applications.

        ('dummy1', np.int16),
        # Reserved

        ('dummy2', np.int16),
        # Reserved

        ('Extra', np.int32, 8),
        # Extra bit values for this event. This array has a fixed length of eight (8)

        ('EventString', 'S', 128),
        # Event string associated with this event record. This string
        # consists of 127 characters plus the required null termination
        # character. If the string is less than 127 characters, the
        # remainder of the characters will be null.
    ])

    def __init__(self, header, records):
        self.header = header
        self.records = records

    @classmethod
    def load_nev(cls, file_path):
        """load a Neuralynx .nev event file and extract the contents"""

        header = NeuralynxBaseLoader.load_header(file_path, NEVLoader._RECORD)

        with open(file_path, 'rb') as fid:
            records = NeuralynxBaseLoader.read_records(fid, NEVLoader._RECORD)

        return cls(header, records)

    def to_df(self):
        data = {
            name: self.records[name]
            for name in self.records.dtype.fields.keys()
            if self.records[name].ndim == 1
        }
        return pd.DataFrame(data)


class MultiNCSLoader(io_common.MultiDataLoader):

    # noinspection PyUnresolvedReferences
    def __init__(self, loaders: dict):
        super().__init__(loaders)

        assert (self.channels['RecordSize'] == 1044).all()
        assert self.channels['sample_count'].nunique() == 1
        assert self.channels['record_count'].nunique() == 1
        assert self.channels['sampling_period'].nunique() == 1

    def get_first_timestamp(self):
        first_timestamps = pd.Series({
            ch: loader.get_first_timestamp()
            for ch, loader in self.loaders.items()
        })
        if first_timestamps.nunique() > 1:
            logging.warning(
                f'Found {first_timestamps.nunique()} first timestamps: {first_timestamps.unique()}'
            )

        return first_timestamps.min()

    def get_joint_header(self) -> pd.DataFrame:
        """Return the header for all the loaders"""
        joint_header = pd.DataFrame.from_dict({
            ch: loader.header
            for ch, loader in self.loaders.items()
        },
            orient='index',
        )
        joint_header.rename_axis(columns='header', index='channel')
        return joint_header

    def get_timestamp_start(self, align_duration=False):
        hdr = self.get_joint_header()
        assert timeslice.to_ms(hdr['time_created'].diff().max()) <= timeslice.ms(seconds=1)
        time_created = hdr['time_created'].mean()

        if align_duration:

            assert timeslice.to_ms(hdr['time_closed'].diff().max()) <= timeslice.ms(seconds=1)
            time_closed = hdr['time_closed'].mean()

            timestamp_duration_ms = timeslice.to_ms(time_closed - time_created)

            if (timestamp_duration_ms - self.duration_ms) > 0:
                center = time_created + (time_closed - time_created) * .5
                half = datetime.timedelta(milliseconds=self.duration_ms * .5)

                new_time_created = center - half
                new_time_closed = center + half

                logging.warning(
                    f'Timestamp and raw data differ by {timeslice.strf_ms(timestamp_duration_ms - self.duration_ms)}.'
                    f' Centering timestamps from <{time_created} - {time_closed}> to  '
                    f'<{new_time_created} - {new_time_closed}>'
                )

                time_created = new_time_created
                # time_closed = new_time_closed

        return time_created

    @classmethod
    def from_paths(cls, paths: List[str], loader_class=NCSLoader):
        """collect all loaders in a dict with channel id as the key"""

        loaders = {}
        for i, path in enumerate(paths):
            loader = loader_class.from_file(path)

            if loader.header['channel_id'] is not None:
                channel_id = loader.header['channel_id']
            else:
                logging.error('Unknown id for channel file: %s', loader.header['full_path'])
                channel_id = -i

            loaders[channel_id] = loader

        loaders = {k: loaders[k] for k in sorted(loaders.keys())}

        return cls(loaders)

    @staticmethod
    def locate_paths(folder: str, channels=None, name='CSC{channel}.ncs') -> list:
        folder = Path(folder)

        if channels is None:
            channel_paths = list(folder.glob(name.format(channel='*')))

        else:
            if isinstance(channels, int):
                channels = range(1, channels + 1)

            channels = list(channels)

            if 0 in channels:
                logging.warning('Expecting channel idcs starting from 1')

            channel_paths = [
                Path(os.path.join(folder, name.format(channel=channel)))
                for channel in channels
            ]

        # sometimes we get dummy files like "CSC1_0001.ncs" that we want to ignore
        channel_paths = [
            p
            for p in channel_paths
            if re.match(r'^CSC\d+\.ncs$', p.name) is not None
        ]

        return channel_paths

    @classmethod
    def from_folder(cls, folder, channels=None, name='CSC{channel}.ncs', loader_class=NCSLoader):
        channel_paths = MultiNCSLoader.locate_paths(folder, channels, name)
        return cls.from_paths(channel_paths, loader_class=loader_class)
