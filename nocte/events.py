"""
Data container for generic LFP events that have a start/stop/reference time.
Internally stored as a simple pd.DataFrame.
"""
import logging

import numpy as np
import pandas as pd
import scipy.interpolate

from nocte import plot as splot
from nocte import timeslice, stacks
from nocte.timeslice import S_TO_MS


def interpolate_trace(data: pd.Series, times: np.array, kind='linear') -> pd.Series:
    lerp = scipy.interpolate.interp1d(
        data.index,
        data.values,
        bounds_error=False,
        fill_value=np.nan,
        kind=kind,
    )

    return pd.Series(lerp(times), index=times)


def interpolate_traces(data: pd.DataFrame, times: np.array, kind='linear') -> pd.DataFrame:
    df = {
        i: interpolate_trace(trace, times, kind=kind)
        for i, (col, trace) in enumerate(data.items())
    }

    df = pd.DataFrame(df)

    df.columns = data.columns

    return df


def interpolate_trace_wrapped(data: pd.Series, times: np.array, kind='linear', period=2 * np.pi) -> pd.Series:
    """Same as above but taking into account that "data" is circular
    (for example the phase of a signal)
    """

    def wrap(trace):
        return (np.unwrap(trace) + (period * .5)) % period - (period * .5)

    data_unwrapped = pd.Series(np.unwrap(data.values, period=period), index=data.index)

    lerp = scipy.interpolate.interp1d(
        data_unwrapped.index,
        data_unwrapped.values,
        bounds_error=False,
        fill_value=np.nan,
        kind=kind,
    )

    new = lerp(times)

    new_wrapped = wrap(new)

    return pd.Series(new_wrapped, index=times)


class Events:
    """
    This class acts mostly as a wrapper around a dataframe registry containing
    metadata about LFP events.

    An event must have a ref_time column and, optionally, a start_time and stop_time.
    """

    def __init__(self, reg: pd.DataFrame):
        self.reg: pd.DataFrame = reg.rename_axis(index='event_id')
        assert self.reg.index.is_unique

    @classmethod
    def from_hdf(cls, path, desc=None):
        # noinspection PyTypeChecker
        return cls(pd.read_hdf(path, desc=desc))

    def to_hdf(self, path, desc=None):
        # noinspection PyTypeChecker
        return self.reg.to_hdf(path, key=desc)

    def _time_cols(self) -> pd.Index:
        return self.reg.columns.intersection(['ref_time', 'start_time', 'stop_time'])

    def _time_cols_param(self, cols, strip=False):
        if cols is None:
            cols = self._time_cols()

        if isinstance(cols, str):
            cols = [cols]

        if strip:
            cols = [c.replace('_time', '') for c in cols]

        return cols

    def round(self, cols=None, decimals=0):
        cols = self._time_cols_param(cols)

        reg = self.reg.copy()

        for col in cols:
            reg[col] = np.round(reg[col], decimals=decimals)

        return self.__class__(reg)

    def __getitem__(self, item):
        return self.reg.__getitem__(item)

    def __setitem__(self, key, value):
        return self.reg.__setitem__(key, value)

    def __len__(self):
        return len(self.reg)

    def __str__(self):
        return self.reg.__str__()

    def __repr__(self):
        return self.reg.__repr__()

    def to_wins(self):
        wins = self.reg[['start_time', 'ref_time', 'stop_time']].copy()
        wins.columns = ['start', 'ref', 'stop']
        return timeslice.Windows(wins)

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.reg._repr_html_()

    def sel_mask(self, mask):
        return self.__class__(self.reg.loc[mask])

    def sel_between(self, **kwargs):
        mask = np.array([self.reg[key].between(*vrange).values for key, vrange in kwargs.items()])
        mask = np.all(mask, axis=0)
        return self.sel_mask(mask)

    def sel(self, **kwargs):
        masks = [
            np.ones(len(self), dtype=bool)
        ]

        for key, value in kwargs.items():
            masks.append((self.reg[key] == value))

        mask = np.all(masks, axis=0)

        return self.sel_mask(mask)

    def sample(self, *args, **kwargs):
        return self.__class__(
            self.reg.sample(*args, **kwargs)
        )

    def sample_uniformly(self, on, count=10, jitter=0):
        values = self.reg[on]

        refs = np.linspace(values.min(), values.max(), count)

        indices = np.argmin(np.abs(values.values[:, np.newaxis] - refs[np.newaxis, :]), axis=0)

        max_idx = len(self) - 1

        indices = np.unique(np.clip(indices + jitter, 0, max_idx))

        indices = np.unique(indices)

        return self.sel_mask(values.index[indices])

    def sort_values(self, *args, **kwargs):
        return self.__class__(
            self.reg.sort_values(*args, **kwargs)
        )

    def lookup_values(self, name, values: pd.Index, cols=None, by='idx'):
        cols = self._time_cols_param(cols, strip=True)

        new_cols = pd.DataFrame({
            f'{col}_{name}': values[self.reg[f'{col}_{by}'].values]
            for col in cols
        }, index=self.reg.index)

        return self.__class__(pd.concat([self.reg, new_cols], axis=1))

    def lookup_series(self, series: pd.Series, name=None, cols=None, by='idx'):
        cols = self._time_cols_param(cols, strip=True)

        if name is None:
            name = series.name

        assert name is not None

        new_cols = pd.DataFrame({
            f'{col}_{name}': series.reindex(self.reg[f'{col}_{by}'].values).values
            for col in cols
        }, index=self.reg.index)

        return self.__class__(pd.concat([self.reg, new_cols], axis=1))

    def lookup_dataframe(self, df: pd.DataFrame):
        ev = self

        for name, series in df.items():
            ev = ev.lookup_series(series)

        return ev

    def lookup_values_per_channel(self, name, stack, cols=None, by='idx'):
        cols = self._time_cols_param(cols, strip=True)

        assert stack.ndim == 2
        assert 'channel' in stack.dims

        all_new_cols = []
        for chan, sreg in self.reg.groupby('channel'):
            trace = stack.sel(channel=chan)

            new_cols = pd.DataFrame(
                {f'{col}_{name}': trace.values[sreg[f'{col}_{by}'].values] for col in cols},
                index=sreg.index
            )

            all_new_cols.append(new_cols)

        all_new_cols = pd.concat(all_new_cols, axis=0, ignore_index=False, sort=True)

        return self.__class__(
            pd.concat([self.reg, all_new_cols], axis=1)
        )

    def lookup_values_interp(self, name, data: pd.Series, cols=None, kind='linear'):
        cols = self._time_cols_param(cols, strip=True)

        if isinstance(cols, str):
            cols = [cols]

        new_cols = pd.DataFrame({
            f'{col}_{name}': interpolate_trace(data, self.reg[f'{col}_time'], kind=kind).values
            for col in cols
        }, index=self.reg.index)

        joint = pd.concat([self.reg, new_cols], axis=1)
        assert joint.columns.is_unique

        return self.__class__(
            joint
        )

    def combine(self, other):
        return self.__class__(
            pd.concat([self.reg, other.reg], axis=0, sort=True, ignore_index=True)
        )

    def get_time_to_closest(self, stop='stop', start='start', sortby='ref') -> pd.Series:

        evs = self.sort_values(f'{sortby}_time')

        to_next = evs.get_inter_event_intervals(stop, start, sortby=sortby)

        df = pd.DataFrame({
            'next': to_next.abs(),
            'prev': to_next.shift(1),
        })

        times = df.min(axis=1).reindex(self.reg.index)

        bad = np.count_nonzero(times < 0)
        if bad > 0:
            logging.warning(f'{bad:,g} events with negative times to next. Events overlap?')

        return times

    def get_time_to_closest_per_channel(self, per='channel', **kwargs) -> pd.Series:

        ts = []
        for ch in self['channel'].unique():
            ts.append(self.sel(**{per: ch}).get_time_to_closest(**kwargs))

        ts = pd.concat(ts)

        assert len(ts) == len(self)
        return ts.reindex(self.reg.index)

    def get_inter_event_intervals(self, first='ref', second='ref', sortby='ref', ascending=True) -> pd.Series:

        reg = self.reg.sort_values(f'{sortby}_time', ascending=ascending)
        td = reg[f'{second}_time'].values[1:] - reg[f'{first}_time'].values[:-1]
        td = pd.Series(td, reg.index[:-1])

        return td.reindex(self.reg.index)

    def get_inter_event_intervals_between_channels(
            self,
            first='ref', second='ref', sortby='ref',
            first_ch=0, second_ch=1
    ) -> pd.Series:

        reg = self.reg.sort_values(f'{sortby}_time')
        reg = reg.loc[reg['channel'].isin([first_ch, second_ch])]

        consecutive = (
                (reg['channel'].values[:-1] == first_ch) &
                (reg['channel'].values[1:] == second_ch)
        )
        first_idcs = reg.index[:-1][consecutive]
        second_idcs = reg.index[1:][consecutive]

        assert len(first_idcs) == len(second_idcs)

        td = reg.loc[second_idcs, f'{second}_time'].values - reg.loc[first_idcs, f'{first}_time'].values

        return pd.Series(td, first_idcs)

    def get_counts_in_bins(self, bins, by='ref_time') -> pd.Series:

        counts, edges = np.histogram(self.reg[by], bins)

        # noinspection PyUnresolvedReferences
        counts = pd.Series(
            counts,
            index=pd.IntervalIndex.from_breaks(edges),
        )

        return counts

    def get_counts(self, load_win, step=1, by='ref_time') -> pd.Series:
        return self.get_counts_in_bins(
            np.arange(*load_win, step),
            by=by,
        )

    def get_histogram_sliding(self, col='amplitude', tbins=None, vbins=None, by='ref_time'):

        if tbins is None:
            global_tbin = timeslice.Win(self[by].min(), self[by].max())
            tbins = global_tbin.arange(global_tbin.length / 100)

        if vbins is None:
            vbins = np.linspace(self[col].min(), self[col].max(), 101)

        hists = []

        for t0, t1 in zip(tbins[:-1], tbins[1:]):
            sel = self.sel_between(**{by: (t0, t1)})
            h, _ = np.histogram(sel[col], bins=vbins)
            hists.append(h)

        hists = pd.DataFrame(
            hists,
            index=pd.IntervalIndex.from_breaks(tbins),
            columns=pd.IntervalIndex.from_breaks(vbins),
        ).T

        return hists

    @staticmethod
    def _get_rate_from_counts(counts, rolling_win=100, win_type='hamming'):
        # noinspection PyTypeChecker
        index: pd.IntervalIndex = counts.index

        rates = counts.values * S_TO_MS / index.length

        rates = pd.Series(rates, index=index.mid)

        if rolling_win is not None:
            rates = rates.rolling(window=rolling_win, center=True, win_type=win_type).mean()

        return rates

    def get_rate(self, load_win, step=1, rolling_win=100, win_type='hamming') -> pd.Series:
        counts = self.get_counts(load_win, step=step)
        return self.__class__._get_rate_from_counts(counts, rolling_win=rolling_win, win_type=win_type)

    def get_rate_in_bins(self, bins, rolling_win=100, win_type='hamming'):
        counts = self.get_counts_in_bins(bins)
        return self.__class__._get_rate_from_counts(counts, rolling_win=rolling_win, win_type=win_type)

    def shift_time(self, shift, cols=None):
        cols = self._time_cols_param(cols)

        reg = self.reg.copy()
        cols = list(cols)
        reg[cols] = reg[cols] + shift
        return self.__class__(reg)

    def mean_roll_by(self, valid_win=None, on='amplitude', by='ref_time', sliding_win=1_000, step=10, nmin=1):
        if valid_win is None:
            valid_win = (self.reg[by].min(), self.reg[by].max())

        return timeslice.mean_roll_by(
            self.reg[by],
            self.reg[on],
            timeslice.Win.build_centered(0, sliding_win),
            np.arange(*valid_win, step),
            nmin=nmin,
        )

    def shuffle(self):
        return self.__class__(
            self.reg.sample(frac=1, replace=False)
        )

    def plot_traces_highlighted(self, ax, traces: stacks.Stack):
        """
        Plot the given multi-channel traces with the detected events highlighted.
        This can be slow if there are many events.
        """
        assert len(traces.shape) == 2

        main_win = traces.get_rel_win()

        for ch in traces.coords['channel']:
            styles = {
                'event': dict(color=splot.COLORS[f'ch{ch}_dark'], linewidth=1, alpha=1),
                'other': dict(color=splot.COLORS[f'ch{ch}'], linewidth=.4, alpha=1),
            }

            wins = self.sel(channel=ch).to_wins()
            wins['cat'] = 'event'
            wins = wins.complement(cat='other', start=main_win.start, stop=main_win.stop)

            trace = traces.sel(channel=ch).to_series()

            splot.plot_trace_highlighted(ax, trace, wins, styles)
