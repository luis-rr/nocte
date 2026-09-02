from __future__ import annotations

import abc
import collections.abc
import warnings

import numpy as np
import pandas as pd

from nocte._coll.windows import Win
from nocte._core.sampling import SamplingRate, TimeGrid

SignalIds = int | collections.abc.Sequence[int] | np.ndarray | pd.Index


class DataLoader(abc.ABC):
    """Access continuous signals stored outside memory."""

    @property
    @abc.abstractmethod
    def sample_count(self) -> int:
        """Number of samples available from the start of the recording."""

    @property
    @abc.abstractmethod
    def sampling(self) -> SamplingRate:
        """Sampling rate used to interpret the stored sample axis."""

    @property
    @abc.abstractmethod
    def signals(self) -> pd.DataFrame:
        """
        Description of independently loadable scalar signals.

        Rows are indexed by unique integer signal IDs under the ``signal``
        identity namespace. Columns are loader-specific descriptive metadata.
        """

    @abc.abstractmethod
    def _load_samples(
        self,
        signals: pd.Index,
        start: int,
        stop: int,
        *,
        adjust_gain: bool,
    ) -> np.ndarray:
        """
        Load an already validated half-open sample range.

        Implementations return shape ``(len(signals), stop - start)`` and must
        preserve the requested signal order.
        """

    @property
    def grid(self) -> TimeGrid:
        """Complete sampling grid exposed by this loader."""
        return TimeGrid(
            sampling=self.sampling,
            start=0.0,
            n_samples=self.sample_count,
        )

    @property
    def duration_ms(self) -> float:
        """Exclusive temporal end of the exposed recording in milliseconds."""
        return self.grid.stop

    def _signal_ids(self, signals: SignalIds) -> pd.Index:
        """Normalize and validate public signal IDs without changing order."""
        if isinstance(signals, (int, np.integer)):
            ids = pd.Index([int(signals)], name='signal')
        else:
            ids = pd.Index(signals, name='signal')

        if len(ids) == 0:
            raise ValueError('at least one signal ID is required')
        if ids.hasnans:
            raise ValueError('signal IDs must not contain missing values')
        if not ids.is_unique:
            raise ValueError('signal IDs must be unique')
        if not pd.api.types.is_integer_dtype(ids.dtype):
            raise TypeError('signal IDs must be integers')

        signal_index = self.signals.index
        if signal_index.name != 'signal':
            raise RuntimeError("loader signal index must be named 'signal'")
        if signal_index.hasnans or not signal_index.is_unique:
            raise RuntimeError('loader signal index must be unique and non-null')
        if not pd.api.types.is_integer_dtype(signal_index.dtype):
            raise RuntimeError('loader signal index must contain integers')

        missing = ids[~ids.isin(signal_index)]
        if len(missing):
            raise KeyError(f'unknown signal IDs: {missing.tolist()}')

        return pd.Index(ids.to_numpy(dtype=np.intp), name='signal')

    def load_samples(
        self,
        signals: SignalIds,
        start: int = 0,
        stop: int | None = None,
        *,
        adjust_gain: bool = True,
    ) -> np.ndarray:
        """Load signals over the half-open sample interval ``[start, stop)``."""
        ids = self._signal_ids(signals)

        if not isinstance(start, (int, np.integer)):
            raise TypeError('start must be an integer sample index')
        if stop is not None and not isinstance(stop, (int, np.integer)):
            raise TypeError('stop must be an integer sample index or None')

        start = int(start)
        stop = self.sample_count if stop is None else int(stop)

        if start < 0:
            raise ValueError('start must be non-negative')
        if stop < start:
            raise ValueError('stop must not precede start')
        if stop > self.sample_count:
            raise ValueError(
                f'sample range [{start}, {stop}) exceeds recording length '
                f'of {self.sample_count} samples'
            )

        values = np.asarray(
            self._load_samples(
                ids,
                start,
                stop,
                adjust_gain=adjust_gain,
            )
        )

        expected_shape = (len(ids), stop - start)
        if values.shape != expected_shape:
            raise RuntimeError(
                f'loader returned shape {values.shape}, expected {expected_shape}'
            )

        return values

    def load(
        self,
        signals: SignalIds,
        win: Win | None = None,
        *,
        adjust_gain: bool = True,
    ) -> tuple[np.ndarray, TimeGrid]:
        """
        Load signals over a millisecond window.

        The requested half-open interval is snapped inward to actual samples on
        this loader's grid. No interpolation or resampling is performed. The
        returned ``TimeGrid`` describes the samples that were actually loaded.
        """
        if win is None:
            grid = self.grid
            start = 0
        else:
            if not isinstance(win, Win):
                raise TypeError('win must be a Win or None')

            grid = TimeGrid.from_aligned_bounds(
                sampling=self.sampling,
                start=win.time_at('start'),
                stop=win.time_at('stop'),
                anchor=0.0,
            )
            start = self.sampling.ms_to_samples_exact(
                grid.start,
                anchor=0.0,
            )

        stop = start + grid.n_samples

        if start < 0 or stop > self.sample_count:
            raise ValueError(
                f'load interval [{grid.start:g}, {grid.stop:g}) ms lies outside '
                f'recording [0, {self.duration_ms:g}) ms'
            )

        values = self.load_samples(
            signals,
            start=start,
            stop=stop,
            adjust_gain=adjust_gain,
        )
        return values, grid


class MultiDataLoader(DataLoader):
    """
    Combine compatible child loaders behind one global signal namespace.

    Child streams are aligned by sample index. Small sampling-rate differences
    are represented by their mean rate; differing lengths are truncated to the
    common minimum. Both recoveries are explicit and issue warnings. No clock
    synchronization, interpolation, or resampling is performed.
    """

    def __init__(
        self,
        loaders: collections.abc.Mapping[int, DataLoader],
        *,
        sampling_rtol: float = 1e-4,
        sampling_atol: float = 1e-6,
    ) -> None:
        if not loaders:
            raise ValueError('at least one child loader is required')

        sampling_rtol = float(sampling_rtol)
        sampling_atol = float(sampling_atol)

        if not np.isfinite(sampling_rtol) or sampling_rtol < 0:
            raise ValueError('sampling_rtol must be finite and non-negative')

        if not np.isfinite(sampling_atol) or sampling_atol < 0:
            raise ValueError('sampling_atol must be finite and non-negative')

        self.loaders = dict(loaders)

        for loader in self.loaders.values():
            if not isinstance(loader, DataLoader):
                raise TypeError('all child loaders must be DataLoader instances')

        self._signals = self._build_signals()
        self._sampling = self._reconcile_sampling(
            rtol=sampling_rtol,
            atol=sampling_atol,
        )
        self._sample_count = self._reconcile_sample_count()

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def sampling(self) -> SamplingRate:
        return self._sampling

    @property
    def signals(self) -> pd.DataFrame:
        return self._signals

    def _build_signals(self) -> pd.DataFrame:
        """Build stable global IDs and retain child routing information."""
        frames: list[pd.DataFrame] = []

        for loader_id, loader in self.loaders.items():
            signals = loader.signals.copy()
            index = signals.index

            if index.name != 'signal':
                raise ValueError("child signal indices must be named 'signal'")
            if index.hasnans or not index.is_unique:
                raise ValueError('child signal indices must be unique and non-null')
            if not pd.api.types.is_integer_dtype(index.dtype):
                raise TypeError('child signal indices must contain integers')

            reserved = {'loader', 'local_signal'}.intersection(signals.columns)
            if reserved:
                raise ValueError(
                    f'child signal metadata uses reserved columns: {sorted(reserved)}'
                )

            signals.insert(
                0,
                'local_signal',
                index.to_numpy(dtype=np.intp),
            )
            signals.insert(0, 'loader', loader_id)
            frames.append(signals.reset_index(drop=True))

        combined = pd.concat(
            frames,
            axis=0,
            ignore_index=True,
            sort=False,
        )
        combined.index = pd.RangeIndex(len(combined), name='signal')
        return combined

    def _reconcile_sampling(self, *, rtol: float, atol: float) -> SamplingRate:
        rates = np.asarray(
            [loader.sampling.rate for loader in self.loaders.values()],
            dtype=float,
        )
        effective_rate = float(np.mean(rates))

        if not np.allclose(
            rates,
            effective_rate,
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(
                'child sampling rates are not compatible: '
                f'{rates.min():g} to {rates.max():g} Hz'
            )

        if np.any(rates != rates[0]):
            warnings.warn(
                'Child sampling rates differ slightly '
                f'({rates.min():g} to {rates.max():g} Hz); '
                f'using mean {effective_rate:g} Hz and aligning by sample index',
                RuntimeWarning,
                stacklevel=3,
            )

        return SamplingRate(effective_rate)

    def _reconcile_sample_count(self) -> int:
        counts = np.asarray(
            [loader.sample_count for loader in self.loaders.values()],
            dtype=np.int64,
        )

        if np.any(counts < 0):
            raise ValueError('child sample counts must be non-negative')

        common_count = int(np.min(counts))
        if np.any(counts != counts[0]):
            warnings.warn(
                'Child sample counts differ '
                f'({int(counts.min()):,d} to {int(counts.max()):,d}); '
                f'using the common first {common_count:,d} samples',
                RuntimeWarning,
                stacklevel=3,
            )

        return common_count

    def _load_samples(
        self,
        signals: pd.Index,
        start: int,
        stop: int,
        *,
        adjust_gain: bool,
    ) -> np.ndarray:
        """Route global signal IDs to child loaders and restore request order."""
        selected = self.signals.loc[
            signals,
            ['loader', 'local_signal'],
        ].copy()

        loader_ids = selected['loader'].to_numpy(dtype=np.intp)
        local_signals = selected['local_signal'].to_numpy(dtype=np.intp)

        rows: list[np.ndarray | None] = [None] * len(selected)

        for loader_id in np.unique(loader_ids):
            positions = np.flatnonzero(loader_ids == loader_id)

            local_ids = pd.Index(
                local_signals[positions],
                name='signal',
            )

            values = self.loaders[int(loader_id)].load_samples(
                local_ids,
                start=start,
                stop=stop,
                adjust_gain=adjust_gain,
            )

            for position, row in zip(positions, values, strict=True):
                rows[int(position)] = row

        if any(row is None for row in rows):
            raise RuntimeError('failed to load all requested signals')

        return np.stack(
            [row for row in rows if row is not None],
            axis=0,
        )
