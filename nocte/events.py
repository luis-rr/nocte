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
from nocte import datadict as dd


@nb.njit(parallel=True)
def _quantile_rolling_nb(
    time: np.ndarray,
    values: np.ndarray,
    new_time: np.ndarray,
    win: tuple,
    q: float,
    nmin: int
) -> np.ndarray:
    """
    Compute the rolling quantile over a time-based window, using a time-sorted array.

    Parameters
    ----------
    time : np.ndarray
        1D sorted array of timestamps (sorted once globally).
    values : np.ndarray
        1D array of values corresponding to time_s, in the same sorted order.
    new_time : np.ndarray
        The times at which we want the window centered.
    win : tuple (left, right)
        The half-window offsets. The window around new_time[i] covers
        [new_time[i] + win[0], new_time[i] + win[1]].
    q : float
        The quantile to compute (e.g. 0.5 for median).
    nmin : int
        Minimum number of points required to return a valid quantile.

    Returns
    -------
    np.ndarray
        Same shape as new_time. Each entry is the rolling quantile for that window
        (or NaN if fewer than nmin data points lie in the window).
    """
    idx_sort = np.argsort(time)
    time = time[idx_sort]
    values = values[idx_sort]

    result = np.empty(new_time.shape[0])

    for i in nb.prange(new_time.shape[0]):
        center = new_time[i]
        start = center + win[0]
        stop  = center + win[1]

        # Find the left and right indices for this window
        left_idx  = np.searchsorted(time, start, side='left')
        right_idx = np.searchsorted(time, stop, side='right')

        # Slice the relevant values
        window_vals = values[left_idx:right_idx]
        if window_vals.size >= nmin:
            result[i] = np.nanquantile(window_vals, q)
        else:
            result[i] = np.nan

    return result


@nb.njit(parallel=True)
def _mean_rolling_nb(
        time: np.ndarray,
        values: np.ndarray,
        new_time: np.ndarray,
        win: tuple,
        nmin: int
) -> np.ndarray:
    """Same as _quantile_rolling_nb but for mean"""
    idx_sort = np.argsort(time)
    time = time[idx_sort]
    values = values[idx_sort]

    result = np.empty(new_time.shape[0])

    for i in nb.prange(new_time.shape[0]):
        center = new_time[i]
        start = center + win[0]
        stop = center + win[1]

        # Find the left and right indices for this window
        left_idx = np.searchsorted(time, start, side='left')
        right_idx = np.searchsorted(time, stop, side='right')

        # Slice the relevant values
        window_vals = values[left_idx:right_idx]

        if window_vals.size >= nmin:
            result[i] = np.nanmean(window_vals)
        else:
            result[i] = np.nan

    return result


@nb.njit(parallel=True)
def _count_rolling_nb(
        time: np.ndarray,
        new_time: np.ndarray,
        win: tuple,
) -> np.ndarray:
    """Same as _quantile_rolling_nb but for counting"""
    idx_sort = np.argsort(time)
    time = time[idx_sort]

    result = np.empty(new_time.shape[0])

    for i in nb.prange(new_time.shape[0]):
        center = new_time[i]
        start = center + win[0]
        stop = center + win[1]

        # Find the left and right indices for this window
        left_idx = np.searchsorted(time, start, side='left')
        right_idx = np.searchsorted(time, stop, side='right')

        # Slice the relevant values
        result[i] = len(time[left_idx:right_idx])

    return result


@nb.njit(parallel=True)
def _rate_gauss_kernel_nb(
        time: np.ndarray,
        new_time: np.ndarray,
        sigma: float,
        width: float = 5.0,
) -> np.ndarray:
    """
    Estimate instantaneous rate using a Gaussian kernel.

    time : sorted event times
    new_time : where to evaluate the rate
    sigma : std of kernel
    width : half-width of kernel support in units of sigma
    """
    idx_sort = np.argsort(time)
    time = time[idx_sort]

    result = np.empty(new_time.shape[0])

    norm = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)

    h = width * sigma

    for i in nb.prange(new_time.shape[0]):
        center = new_time[i]
        start = center - h
        stop = center + h

        left_idx = np.searchsorted(time, start, side='left')
        right_idx = np.searchsorted(time, stop, side='right')

        acc = 0.0
        for j in range(left_idx, right_idx):
            u = center - time[j]
            acc += norm * np.exp(-0.5 * (u / sigma) ** 2)

        result[i] = acc

    return result


def interpolate_trace(data: pd.Series, times: np.ndarray) -> pd.Series:

    values = np.interp(
        times,
        data.index,
        data.values,
        left=np.nan,
        right=np.nan,
    )

    return pd.Series(values, index=times)


def interpolate_traces(data: pd.DataFrame, times: np.ndarray) -> pd.DataFrame:
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
        reg: pd.DataFrame = reg.rename_axis(index='event_id')
        assert reg.index.is_unique
        super().__init__(reg)

    @classmethod
    def from_data_dict(cls, data: dd.DataDict):
        """
        Flatten a DataDict[Events] into a single Events object.
        """
        joint_reg = []

        for k, events in data.items():
            reg = events.reg.copy()

            row = data.reg.loc[k]

            for col, value in row.items():
                reg[col] = value

            joint_reg.append(reg)

        joint_reg = pd.concat(joint_reg, axis=0, ignore_index=True)

        return cls(joint_reg)

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

        wins.rename(
            columns=dict(
                start_time='start',
                stop_time='stop',
                ref_time='ref',
            ),
            inplace=True
        )

        return timeslice.Windows(wins)

    def to_wins_around(self, win_ms, col='ref_time'):
        """
        Creates fixed-sized windows around these events.
        """
        wins = self.reg.copy()

        if 'start_time' in wins.columns:
            logging.warning(f'Overwriting event column "start_time"')

        wins['start_time'] = wins[col] + win_ms[0]

        if 'stop_time' in wins.columns:
            logging.warning(f'Overwriting event column "start_time"')

        wins['stop_time'] = wins[col] + win_ms[1]

        wins.rename(
            columns=dict(
                start_time='start',
                stop_time='stop',
                ref_time='ref',
            ),
            inplace=True
        )

        return timeslice.Windows(wins)

    def _repr_html_(self):
        # noinspection PyProtectedMember,PyCallingNonCallable
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
        values = interpolate_trace(trace, self.reg[col].values).values
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

        # noinspection PyTypeChecker,PyNoneFunctionAssignment,PyArgumentList
        evs: Events = self.sort_values(f'{sortby}_time')

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

    def get_histogram2d(self, col='amplitude', tbins=None, vbins=None, by='ref_time'):

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

    def shift_time(self, ts: float | pd.Series | np.ndarray, cols=None, copy=True):

        if isinstance(ts, (float, int, np.ndarray)):
            ts = pd.Series(ts, index=self.index)

        if not ts.index.equals(self.index):
            raise ValueError("shift_time Series must be indexed like self")

        ts = ts.reindex(self.index).values

        cols = self._time_cols_param(cols)

        if copy:
            reg = self.reg.copy()
        else:
            reg = self.reg

        cols = list(cols)
        reg[cols] = reg[cols] + ts[:, np.newaxis]
        return self.__class__(reg)

    def count_rolling(
            self,
            valid_win: timeslice.Win = None,
            by='ref_time',
            sliding_win=1_000,
            step=10,
        ) -> pd.Series:
        """Calculate how many items are in a sliding window over time (by)"""
        valid_win = self.get_global_win(by) if valid_win is None else valid_win

        time = self.reg[by].values

        win = timeslice.Win.build_centered(0, sliding_win)
        new_time = valid_win.arange(step)

        res = _count_rolling_nb(
            time=time,
            win=tuple(win),
            new_time=new_time,
        )

        return pd.Series(res, index=new_time)

    def rate_rolling_gauss(
            self,
            sigma: float,
            valid_win: timeslice.Win = None,
            by='ref_time',
            step=1_000,
        ) -> pd.Series:
        """Estaimate instantaneous rate by convolving with a gaussian kernel. Better for low rates than counting."""

        valid_win = self.get_global_win(by) if valid_win is None else valid_win

        sampling_t = valid_win.arange(step)

        rate = _rate_gauss_kernel_nb(
            time=self[by].values,
            new_time=sampling_t,
            sigma=sigma,
        )

        return pd.Series(
            rate * timeslice.ms(seconds=1),
            index=sampling_t,
        )

    def rate_rolling_box(
            self,
            valid_win: timeslice.Win = None,
            by='ref_time',
            sliding_win=10_000,
            step=1_000,
        ) -> pd.Series:
        """Estaimate instantaneous rate by counting in a sliding window"""

        counts = self.count_rolling(valid_win=valid_win, by=by, sliding_win=sliding_win, step=step)
        return counts / (sliding_win / timeslice.ms(seconds=1))

    def mean_rolling(
            self,
            valid_win: timeslice.Win = None,
            on='amplitude',
            by='ref_time',
            sliding_win=1_000,
            step=10,
            nmin=1,
        ) -> pd.Series:
        """Calculate the mean of a property (on) in a sliding window over time (by)"""
        return self._rolling_nb(
            func=_mean_rolling_nb,
            valid_win=valid_win,
            on=on,
            by=by,
            sliding_win=sliding_win,
            step=step,
            nmin=nmin,
        )

    def quantile_rolling(
            self,
            q,
            valid_win: timeslice.Win = None,
            on='amplitude',
            by='ref_time',
            sliding_win=1_000,
            step=10,
            nmin=1,
    ) -> pd.Series:
        """Calculate the quantile of a property (on) in a sliding window over time (by)"""
        return self._rolling_nb(
            func=_quantile_rolling_nb,
            valid_win=valid_win,
            on=on,
            by=by,
            sliding_win=sliding_win,
            step=step,
            nmin=nmin,
            q=q,
        )

    def iqr_rolling(
            self,
            low=0.25,
            mid=0.5,
            high=0.75,
            **kwargs,
    ) -> pd.DataFrame:
        """Calculate the median and inter-quantile range of a property (on) in a sliding window over time (by)"""
        return pd.DataFrame({
            name: self.quantile_rolling(q=q, **kwargs)
            for name, q in dict(low=low, mid=mid, high=high).items()
        })

    def _rolling_nb(
            self,
            func,
            valid_win: timeslice.Win = None,
            on='amplitude',
            by='ref_time',
            sliding_win=1_000,
            step=10,
            nmin=1,
            **kwargs,
    ) -> pd.Series:
        """
        Wrapper for  numba implementation of rolling calculations.
        Generally: for every step, take the mean/quantile/etc of all values "on" whose corresponding "by" falls within
        a given local window.
        """
        valid_win = self.get_global_win(by) if valid_win is None else valid_win

        x = self.reg[by].values
        y = self.reg[on].values

        win = timeslice.Win.build_centered(0, sliding_win)
        xvals = valid_win.arange(step)

        res = func(
            time=x,
            values=y,
            win=tuple(win),
            new_time=xvals,
            nmin=nmin,
            **kwargs,
        )

        return pd.Series(res, index=xvals)

    def locate_within(self, wins: timeslice.Windows, by='exp_name', on='ref_time') -> pd.DataFrame:
        """calculate the relative time of the events within windows after matching them by a column"""
        result = []

        for by_val, wins_sel in wins.iter_groupby(by):
            events_sel = self.sel(**{by: by_val})

            if len(events_sel) > 0:
                classified: pd.DataFrame = wins_sel.classify_events(events_sel[on])

                result.append(classified)

        if len(result) > 0:
            result = pd.concat(result)

        else:
            result = pd.DataFrame(dict(
                win_idx=pd.Series(dtype='int'),
                delay=pd.Series(dtype='float'),
                cat=pd.Series(dtype='str'),
            ))

        result = result.reindex(self.index)

        return result

    def classify_by(self, wins: timeslice.Windows, by='exp_name', on='ref_time', col='cat') -> pd.Series:
        """classify each event by the category of selected windows after matching them by a column"""
        return self.locate_within(wins, by=by, on=on)[col]

    def crop(self, win: timeslice.Win, on='ref_time'):
        """
        Extract the events within windows after matching them by a column.
        Events in no window are dropped.
        """
        return self.sel_between(**{on: win})

    def extract(self, wins: timeslice.Windows, by='exp_name', on='ref_time', align=None):

        locs = self.locate_within(wins, by=by, on=on)

        locs = locs.dropna()

        reg = self.reg.loc[locs.index].copy()

        reg['win_idx'] = locs['win_idx'].astype(wins.index.dtype)

        extracted = self.__class__(reg)

        if align is not None:
            refs = wins.relative_time(align)
            shifts = extracted['win_idx'].map(refs).values
            extracted = extracted.shift_time(-1 * shifts)

        return extracted

    def sel_time(self, win: timeslice.Win, on='ref_time', reset=None):  # TODO deprecate
        sel = self.crop(win, on=on)

        if reset is not None:
            sel = sel.shift_time(-win.relative_time('start'))

        return sel

    def get_global_win(self, by='ref_time') -> timeslice.Win:
        return timeslice.Win(self.reg[by].min(), self.reg[by].max())

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

