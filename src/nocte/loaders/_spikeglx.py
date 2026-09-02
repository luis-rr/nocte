from __future__ import annotations

import collections.abc
import pathlib
import re
import typing
import warnings

import numpy as np
import pandas as pd

import nocte.loaders._core
from nocte._core.sampling import SamplingRate

MICROVOLTS_PER_VOLT = 1_000_000.0
NP2_TYPES = {21, 24, 2003, 2013, 2020}


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


def _series_raw(series: pd.Series) -> collections.abc.Mapping[str, str]:
    value = _series_value(series, 'raw')
    if not isinstance(value, collections.abc.Mapping):
        raise TypeError("'raw' metadata must be a mapping")

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError("'raw' metadata must map strings to strings")
        result[key] = item
    return result


def _series_imro(
    series: pd.Series,
) -> collections.abc.Mapping[int, tuple[str, ...]]:
    value = _series_value(series, 'imro')
    if not isinstance(value, collections.abc.Mapping):
        raise TypeError("'imro' metadata must be a mapping")

    result: dict[int, tuple[str, ...]] = {}
    for key, item in value.items():
        if not isinstance(key, (int, np.integer)):
            raise TypeError('IMRO keys must be integers')
        if not isinstance(item, tuple) or not all(
            isinstance(part, str) for part in item
        ):
            raise TypeError('IMRO rows must be tuples of strings')
        result[int(key)] = item
    return result


def _series_counts(series: pd.Series) -> tuple[int, int, int]:
    value = _series_value(series, 'acquired_counts')
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("'acquired_counts' must be a length-three tuple")

    counts: list[int] = []
    for item in value:
        if not isinstance(item, (int, np.integer)):
            raise TypeError("'acquired_counts' values must be integers")
        counts.append(int(item))
    return counts[0], counts[1], counts[2]


def _series_saved_channels(series: pd.Series) -> np.ndarray:
    value = _series_value(series, 'saved_channels')
    array = np.asarray(value)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("'saved_channels' must be a one-dimensional integer array")
    return array.astype(np.int64, copy=False)


def _index_first_int(index: pd.Index) -> int:
    values = index.to_numpy(dtype=np.intp)
    if len(values) != 1:
        raise ValueError(f'expected exactly one index value, found {len(values)}')
    return int(values[0])


def _read_meta_raw(path: str | pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}

    with pathlib.Path(path).open(encoding='utf-8') as file:
        for number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            key, sep, value = line.partition('=')
            if not sep:
                raise ValueError(f'Malformed SpikeGLX metadata line {number}')
            values[key.lstrip('~')] = value

    return values


def _parse_subset(value: str, count: int) -> np.ndarray:
    if value in {'all', '*'}:
        channels = np.arange(count, dtype=np.int64)
    else:
        parsed: list[int] = []

        for part in value.split(','):
            if ':' in part:
                start, stop = (int(x) for x in part.split(':', maxsplit=1))
                if stop < start:
                    raise ValueError(f'Invalid saved-channel range {part!r}')
                parsed.extend(range(start, stop + 1))
            else:
                parsed.append(int(part))

        channels = np.asarray(parsed, dtype=np.int64)

    if len(channels) != count:
        raise ValueError(
            f'snsSaveChanSubset gives {len(channels)} channels, expected {count}'
        )
    if len(np.unique(channels)) != len(channels) or np.any(channels < 0):
        raise ValueError('Saved SpikeGLX channels must be unique and non-negative')

    return channels


def _parse_counts(value: str) -> tuple[int, int, int]:
    try:
        counts = tuple(int(x) for x in value.split(','))
    except ValueError as error:
        raise ValueError(f'Could not parse channel counts {value!r}') from error

    if len(counts) != 3 or any(x < 0 for x in counts):
        raise ValueError(f'Expected three non-negative channel counts, got {value!r}')

    return counts[0], counts[1], counts[2]


def _parse_imro(value: str | None) -> dict[int, tuple[str, ...]]:
    if not value:
        return {}

    groups = re.findall(r'\(([^()]*)\)', value)
    entries: dict[int, tuple[str, ...]] = {}

    for group in groups[1:]:
        fields = tuple(group.replace(',', ' ').split())
        if fields:
            try:
                entries[int(fields[0])] = fields
            except ValueError:
                pass

    return entries


def _probe_type(raw: collections.abc.Mapping[str, str]) -> int | None:
    try:
        return int(raw['imDatPrb_type'])
    except (KeyError, ValueError):
        return None


def _channel_info(
    acquisition_channel: int,
    counts: tuple[int, int, int],
) -> tuple[str, int | None]:
    ap, lf, sy = counts

    if acquisition_channel < ap:
        return 'AP', acquisition_channel
    if acquisition_channel < ap + lf:
        return 'LF', acquisition_channel - ap
    if acquisition_channel < ap + lf + sy:
        return 'SY', None

    raise ValueError(
        f'Acquisition channel {acquisition_channel} lies outside acqApLfSy={counts}'
    )


def _gain(
    raw: collections.abc.Mapping[str, str],
    imro: collections.abc.Mapping[int, tuple[str, ...]],
    band: str,
    channel: int,
) -> float | None:
    entry = imro.get(channel)

    if entry is not None and len(entry) >= 6:
        try:
            gain = float(entry[3 if band == 'AP' else 4])
        except ValueError:
            gain = 0.0
        if gain > 0:
            return gain

    if _probe_type(raw) in NP2_TYPES:
        return 80.0

    return None


def read_meta(
    meta_path: str | pathlib.Path,
    bin_path: str | pathlib.Path | None = None,
) -> pd.Series:
    """Read the SpikeGLX metadata required for direct Neuropixels loading."""
    meta_path = pathlib.Path(meta_path)
    bin_path = (
        meta_path.with_suffix('.bin') if bin_path is None else pathlib.Path(bin_path)
    )
    raw = _read_meta_raw(meta_path)

    if raw.get('typeThis') != 'imec':
        raise ValueError(f'Expected imec metadata, found {raw.get("typeThis")!r}')

    try:
        channel_count = int(raw['nSavedChans'])
        sampling_rate = float(raw['imSampRate'])
        ai_range_max = float(raw['imAiRangeMax'])
    except KeyError as error:
        raise ValueError(
            f'Missing SpikeGLX metadata field {error.args[0]!r}'
        ) from error

    if channel_count <= 0 or sampling_rate <= 0 or ai_range_max <= 0:
        raise ValueError('Invalid SpikeGLX channel count, sampling rate, or AI range')

    saved_channels = _parse_subset(
        raw.get('snsSaveChanSubset', 'all'),
        channel_count,
    )

    acquired = raw.get('acqApLfSy')
    if acquired is None:
        if raw.get('snsSaveChanSubset', 'all') not in {'all', '*'}:
            raise ValueError('acqApLfSy is required for selectively saved data')
        acquired = raw.get('snsApLfSy')
    if acquired is None:
        raise ValueError('Missing acqApLfSy/snsApLfSy')
    acquired_counts = _parse_counts(acquired)

    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)

    file_size = bin_path.stat().st_size
    bytes_per_sample = 2 * channel_count
    if file_size % bytes_per_sample:
        raise ValueError(
            f'Binary size {file_size:,d} is not divisible by '
            f'{bytes_per_sample} bytes/sample'
        )

    expected_size = raw.get('fileSizeBytes')
    if expected_size is not None and int(expected_size) != file_size:
        warnings.warn(
            'SpikeGLX fileSizeBytes differs from the binary file; '
            'using the actual binary size',
            RuntimeWarning,
            stacklevel=2,
        )

    if 'imMaxInt' in raw:
        max_int = float(raw['imMaxInt'])
    else:
        max_int = 512.0
        warnings.warn(
            'SpikeGLX metadata has no imMaxInt; assuming 512 for older NP1 data',
            RuntimeWarning,
            stacklevel=2,
        )

    if max_int <= 0:
        raise ValueError('imMaxInt must be positive')

    return pd.Series(
        {
            'meta_path': meta_path,
            'bin_path': bin_path,
            'raw': raw,
            'channel_count': channel_count,
            'sample_count': file_size // bytes_per_sample,
            'sampling_rate': sampling_rate,
            'saved_channels': saved_channels,
            'acquired_counts': acquired_counts,
            'ai_range_max': ai_range_max,
            'max_int': max_int,
            'imro': _parse_imro(raw.get('imroTbl')),
        },
        dtype=object,
    )


def make_memmap_raw(
    bin_path: str | pathlib.Path,
    channel_count: int,
    sample_count: int,
) -> np.memmap:
    return np.memmap(
        pathlib.Path(bin_path),
        dtype=np.int16,
        mode='r',
        order='F',
        shape=(channel_count, sample_count),
    )


class NeuropixelsLoader(nocte.loaders._core.DataLoader):
    """Loader for one Neuropixels SpikeGLX binary/meta pair."""

    def __init__(self, meta: pd.Series, memmap: np.memmap) -> None:
        self.meta = meta.copy()
        self.memmap = memmap

        self._sample_count = _series_int(self.meta, 'sample_count')
        self._channel_count = _series_int(self.meta, 'channel_count')
        self._sampling = SamplingRate(_series_float(self.meta, 'sampling_rate'))
        self._raw = _series_raw(self.meta)
        self._imro = _series_imro(self.meta)
        self._counts = _series_counts(self.meta)
        self._saved_channels = _series_saved_channels(self.meta)
        self._ai_range_max = _series_float(self.meta, 'ai_range_max')
        self._max_int = _series_float(self.meta, 'max_int')

        expected = (self._channel_count, self._sample_count)
        if memmap.shape != expected:
            raise ValueError(f'Memmap shape {memmap.shape} does not match {expected}')

        self._signals = self._build_signals()

    def _build_signals(self) -> pd.DataFrame:
        base_conversion = self._ai_range_max / self._max_int * MICROVOLTS_PER_VOLT

        rows: list[dict[str, object]] = []
        for stored, acquired_value in enumerate(self._saved_channels):
            acquired = int(acquired_value)
            band, channel = _channel_info(acquired, self._counts)
            is_system = channel is None
            gain = (
                None if channel is None else _gain(self._raw, self._imro, band, channel)
            )

            rows.append(
                {
                    'stored_channel': stored,
                    'acquisition_channel': acquired,
                    'band': band,
                    'channel': channel,
                    'is_system': is_system,
                    'gain': gain,
                    'conversion_uV': (
                        np.nan if gain is None else base_conversion / gain
                    ),
                    'unit': 'raw' if is_system else 'uV',
                }
            )

        signals = pd.DataFrame(rows)
        signals.index = pd.RangeIndex(len(signals), name='signal')
        return signals

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def sampling(self) -> SamplingRate:
        return self._sampling

    @property
    def signals(self) -> pd.DataFrame:
        return self._signals

    def lookup(self, *, channel: int) -> int:
        """Return the local signal ID for one zero-based neural channel."""
        matches = self.signals.index[
            self.signals['channel'].eq(channel) & ~self.signals['is_system']
        ]
        if len(matches) != 1:
            raise KeyError(
                f'Expected one signal for channel={channel}; found {len(matches)}'
            )
        return _index_first_int(matches)

    def _load_samples(
        self,
        signals: pd.Index,
        start: int,
        stop: int,
        *,
        adjust_gain: bool,
    ) -> np.ndarray:
        selected = self.signals.loc[signals]
        stored = selected['stored_channel'].to_numpy(dtype=np.intp)
        raw = np.asarray(self.memmap[stored, start:stop])

        if not adjust_gain:
            return raw

        values = raw.astype(np.float64)
        neural = ~selected['is_system'].to_numpy(dtype=bool)
        factors = selected['conversion_uV'].to_numpy(dtype=float)

        if np.any(neural & ~np.isfinite(factors)):
            missing = selected.index[neural & ~np.isfinite(factors)].tolist()
            raise ValueError(
                f'Unknown voltage conversion for signal IDs {missing}; '
                'use adjust_gain=False'
            )

        values[neural] *= factors[neural, np.newaxis]
        return values

    @classmethod
    def from_spikeglx(
        cls,
        meta_path: str | pathlib.Path,
        bin_path: str | pathlib.Path | None = None,
        *,
        expected_sampling_rate: float | None = None,
    ) -> typing.Self:
        meta = read_meta(meta_path, bin_path)

        if expected_sampling_rate is not None:
            expected_sampling_rate = float(expected_sampling_rate)
            actual = _series_float(meta, 'sampling_rate')

            if expected_sampling_rate <= 0:
                raise ValueError('expected_sampling_rate must be positive')
            if not np.isclose(actual, expected_sampling_rate, rtol=1e-12):
                sample_count = _series_int(meta, 'sample_count')
                duration_change = (
                    sample_count / expected_sampling_rate - sample_count / actual
                ) * 1000.0
                warnings.warn(
                    f'Overriding SpikeGLX sampling rate {actual:g} with '
                    f'{expected_sampling_rate:g} Hz; duration changes by '
                    f'{duration_change:g} ms',
                    RuntimeWarning,
                    stacklevel=2,
                )
                meta['sampling_rate'] = expected_sampling_rate

        memmap = make_memmap_raw(
            _series_path(meta, 'bin_path'),
            _series_int(meta, 'channel_count'),
            _series_int(meta, 'sample_count'),
        )
        return cls(meta, memmap)

    @staticmethod
    def find_sources(
        folder: str | pathlib.Path,
        *,
        band: str = 'ap',
        allow_lf: bool = False,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        folder = pathlib.Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(folder)

        def find(which: str) -> tuple[pathlib.Path, pathlib.Path] | None:
            metas = sorted(folder.glob(f'*.{which}.meta'))
            if not metas:
                return None
            if len(metas) != 1:
                raise FileNotFoundError(
                    f'Expected one *.{which}.meta in {folder}; found {len(metas)}'
                )

            binary = metas[0].with_suffix('.bin')
            if not binary.is_file():
                raise FileNotFoundError(binary)
            return metas[0], binary

        band = band.lower()
        if band not in {'ap', 'lf'}:
            raise ValueError("band must be 'ap' or 'lf'")

        sources = find(band)
        if sources is not None:
            return sources

        if band == 'ap' and allow_lf:
            sources = find('lf')
            if sources is not None:
                warnings.warn(
                    f'No AP file found in {folder}; using LF instead',
                    RuntimeWarning,
                    stacklevel=2,
                )
                return sources

        raise FileNotFoundError(f'No *.{band}.meta/.bin pair found in {folder}')

    @classmethod
    def from_folder(
        cls,
        folder: str | pathlib.Path,
        *,
        band: str = 'ap',
        allow_lf: bool = False,
        expected_sampling_rate: float | None = None,
    ) -> typing.Self:
        meta, binary = cls.find_sources(
            folder,
            band=band,
            allow_lf=allow_lf,
        )
        return cls.from_spikeglx(
            meta,
            binary,
            expected_sampling_rate=expected_sampling_rate,
        )


class MultiProbeLoader(nocte.loaders._core.MultiDataLoader):
    """Recording-level interface over multiple Neuropixels probes."""

    def __init__(
        self,
        loaders: collections.abc.Mapping[int, NeuropixelsLoader],
        *,
        sampling_rtol: float = 1e-4,
        sampling_atol: float = 1e-6,
    ) -> None:
        normalized: dict[int, NeuropixelsLoader] = {}

        for probe, loader in loaders.items():
            if not isinstance(probe, (int, np.integer)) or int(probe) < 0:
                raise ValueError('Probe IDs must be non-negative integers')
            normalized[int(probe)] = loader

        self._probe_loaders = dict(sorted(normalized.items()))
        super().__init__(
            self._probe_loaders,
            sampling_rtol=sampling_rtol,
            sampling_atol=sampling_atol,
        )
        self._signals.insert(0, 'probe', self._signals['loader'].astype(int))

    def lookup(self, *, probe: int, channel: int) -> int:
        """Return the global signal ID for one zero-based probe/channel pair."""
        matches = self.signals.index[
            self.signals['probe'].eq(probe)
            & self.signals['channel'].eq(channel)
            & ~self.signals['is_system']
        ]
        if len(matches) != 1:
            raise KeyError(
                f'Expected one signal for probe={probe}, channel={channel}; '
                f'found {len(matches)}'
            )
        return _index_first_int(matches)

    @staticmethod
    def _probe_id(path: pathlib.Path) -> int | None:
        for text in (path.name, path.parent.name):
            match = re.search(r'(?:^|[._-])imec(\d+)(?:[._-]|$)', text)
            if match is not None:
                return int(match.group(1))

        return 0 if '.imec.' in path.name.lower() else None

    @classmethod
    def find_sources(
        cls,
        folder: str | pathlib.Path,
        *,
        band: str = 'ap',
        allow_lf: bool = False,
    ) -> dict[int, tuple[pathlib.Path, pathlib.Path]]:
        folder = pathlib.Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(folder)

        band = band.lower()
        if band not in {'ap', 'lf'}:
            raise ValueError("band must be 'ap' or 'lf'")

        def find(which: str) -> dict[int, tuple[pathlib.Path, pathlib.Path]]:
            result: dict[int, tuple[pathlib.Path, pathlib.Path]] = {}

            for meta in sorted(folder.rglob(f'*.{which}.meta')):
                probe = cls._probe_id(meta)
                if probe is None:
                    warnings.warn(
                        f'Ignoring {meta}: could not infer probe ID',
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    continue

                binary = meta.with_suffix('.bin')
                if not binary.is_file():
                    warnings.warn(
                        f'Ignoring {meta}: missing {binary.name}',
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    continue

                if probe in result:
                    raise ValueError(
                        f'Multiple {which.upper()} files found for probe '
                        f'{probe}; recording segments are not concatenated '
                        'automatically'
                    )
                result[probe] = (meta, binary)

            return dict(sorted(result.items()))

        sources = find(band)
        if sources:
            return sources

        if band == 'ap' and allow_lf:
            sources = find('lf')
            if sources:
                warnings.warn(
                    f'No AP sources found in {folder}; using LF instead',
                    RuntimeWarning,
                    stacklevel=2,
                )
                return sources

        raise FileNotFoundError(
            f'No Neuropixels {band.upper()} sources found in {folder}'
        )

    @classmethod
    def from_sources(
        cls,
        sources: collections.abc.Mapping[
            int,
            tuple[str | pathlib.Path, str | pathlib.Path],
        ],
        **kwargs: typing.Any,
    ) -> typing.Self:
        if not sources:
            raise FileNotFoundError('No Neuropixels sources were provided')

        loaders = {
            int(probe): NeuropixelsLoader.from_spikeglx(meta, binary)
            for probe, (meta, binary) in sources.items()
        }
        return cls(loaders, **kwargs)

    @classmethod
    def from_folder(
        cls,
        folder: str | pathlib.Path,
        *,
        band: str = 'ap',
        allow_lf: bool = False,
        **kwargs: typing.Any,
    ) -> typing.Self:
        sources = cls.find_sources(folder, band=band, allow_lf=allow_lf)
        return cls.from_sources(sources, **kwargs)
