"""
Data container for generic LFP events that have a start/stop/reference time.
Internally stored as a simple pd.DataFrame.
"""
import functools
import logging

import numba as nb
import numpy as np
import pandas as pd

from nocte import plot as splot
from nocte import timeslice, stacks
from nocte.df_wrapper import DataFrameWrapper
from nocte.timeslice import S_TO_MS


@nb.njit(parallel=True)
def _mean_roll_by_nb(_x: np.array, _y: np.array, _win: tuple, _xvals: np.array, nmin):
    """ fast implementation of mean_roll_by """
    rolled = np.empty(len(_xvals))

    for i in nb.prange(len(_xvals)):
        x_center = _xvals[i]
        vmin = x_center + _win[0]
        vmax = x_center + _win[1]

        sel = _y[(vmin <= _x) & (_x <= vmax)]

        if len(sel) >= nmin:
            rolled[i] = np.nanmean(sel)
        else:
            rolled[i] = np.nan

    return rolled


def _mean_roll_by(x: np.array, y: np.array, win: tuple, xvals: np.array, nmin=1):
    """
    for every value in xvals, take the mean of all ys whose corresponding x falls within a given local window
    """

    if isinstance(x, pd.Series):
        x = x.values

    if isinstance(y, pd.Series):
        y = y.values

    assert len(x) == len(y)

    res = _mean_roll_by_nb(
        np.asarray(x),
        np.asarray(y),
        tuple(win),
        np.asarray(xvals),
        nmin=nmin,
    )

    return pd.Series(res, index=xvals)


def interpolate_trace(data: pd.Series, times: np.array) -> pd.Series:
    values = np.interp(
        times,
        data.index,
        data.values,
        left=np.nan,
        right=np.nan
    )

    return pd.Series(values, index=times)


def interpolate_traces(data: pd.DataFrame, times: np.array) -> pd.DataFrame:
    df = {
        i: interpolate_trace(trace, times)
        for i, (col, trace) in enumerate(data.items())
    }

    df = pd.DataFrame(df)

    df.columns = data.columns

    return df


class Events(DataFrameWrapper):
    """
    This class acts mostly as a wrapper around a dataframe registry containing
    metadata about LFP events.

    An event must have a ref_time column and, optionally, a start_time and stop_time.
    """

    def __init__(self, reg: pd.DataFrame):
        reg = reg.rename_axis(index='event_id')
        assert reg.index.is_unique
        super().__init__(reg)

    @property
    def loc(self):
        """pd.DataFrame accessor"""
        return self.reg.loc

    @property
    def iloc(self):
        """pd.DataFrame accessor"""
        return self.reg.iloc

    @functools.wraps(pd.DataFrame.drop)
    def drop(self, *args, **kwargs):
        return self.__class__(self.reg.drop(*args, **kwargs))

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

    def __len__(self):
        return len(self.reg)

    def to_wins(self):
        wins = self.reg.copy()

        if 'start_time' not in wins.columns:
            wins['start_time'] = wins['ref_time']

        if 'stop_time' not in wins.columns:
            wins['stop_time'] = wins['ref_time']

        wins.rename(columns=dict(start_time='start', stop_time='stop', ref_time='ref'), inplace=True)

        return timeslice.Windows(wins)

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.reg._repr_html_()

    def sample_uniformly(self, on, count=10, jitter=0):
        values = self.reg[on]

        refs = np.linspace(values.min(), values.max(), count)

        indices = np.argmin(np.abs(values.values[:, np.newaxis] - refs[np.newaxis, :]), axis=0)

        max_idx = len(self) - 1

        indices = np.unique(np.clip(indices + jitter, 0, max_idx))

        indices = np.unique(indices)

        return self.sel_mask(values.index[indices])

    def lookup(self, trace: pd.Series, col='ref_time') -> pd.Series:
        """
        Look up the value of a signal at the time of each one of these events.
        If the exact time is not available, it will be generated through linear interpolation.

        Parameters
        ----------
        trace: a time series with time as index and data as values
        col: a column property of these events

        Returns
        -------
        A series with index matching these events and value extracted from the trace.

        """
        values = interpolate_trace(trace, self.reg[col])
        return pd.Series(values, index=self.index)

    def lookup_and_set(self, name, data: pd.Series, by, cols=None):
        """Look up many values at once and return a copy of this object with those values set as new columns"""
        cols = self._time_cols_param(cols, strip=True)

        if isinstance(cols, str):
            cols = [cols]

        if not isinstance(data, pd.Series):
            data = np.asarray(data)
            data = pd.Series(data, index=np.arange(len(data)))

        new_cols = pd.DataFrame({
            f'{col}_{name}': self.lookup(data, col=f'{col}_{by}').values
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

        # noinspection PyTypeChecker
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

    def plot_traces_highlighted(self, ax, traces: stacks.Stack):
        """
        Plot the given multi-channel traces with the detected events highlighted.
        This can be slow if there are many events.
        """
        assert len(traces.shape) == 2

        main_win = traces.get_global_win()

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
