"""
Manage conversion between timescales & sampling rates, as well as defining windows of time
that can be used to cut data.
"""
import functools
import logging
from datetime import timedelta

import numba
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nocte.df_wrapper import DataFrameWrapper, _optional_pbar

S_TO_MS = 1e3
MS_TO_S = 1. / S_TO_MS


@numba.njit(parallel=True)
def _mean_roll_by(_x: np.array, _y: np.array, _win: tuple, _xvals: np.array, nmin):
    """ fast implementation of mean_roll_by """
    rolled = np.empty(len(_xvals))

    for i in numba.prange(len(_xvals)):
        x_center = _xvals[i]
        vmin = x_center + _win[0]
        vmax = x_center + _win[1]

        sel = _y[(vmin <= _x) & (_x <= vmax)]

        if len(sel) >= nmin:
            rolled[i] = np.nanmean(sel)
        else:
            rolled[i] = np.nan

    return rolled


def mean_roll_by(x: np.array, y: np.array, win: tuple, xvals: np.array, nmin=1):
    """
    for every value in xvals, take the mean of all ys whose corresponding x falls within a given local window
    """

    if isinstance(x, pd.Series):
        x = x.values

    if isinstance(y, pd.Series):
        y = y.values

    assert len(x) == len(y)

    res = _mean_roll_by(
        np.asarray(x),
        np.asarray(y),
        tuple(win),
        np.asarray(xvals),
        nmin=nmin,
    )

    return pd.Series(res, index=xvals)


def to_ms(t) -> float:
    """allow parameter to be float (milliseconds) or fancier timedelta objects"""
    if isinstance(t, timedelta):
        t = t.total_seconds() * S_TO_MS

    if np.issubdtype(type(t), np.integer):
        t = int(t)

    if np.issubdtype(type(t), np.floating):
        t = float(t)

    return t


def ms(**kwargs) -> float:
    """short-hand to write a millisecond time-stamp using hours=, minutes=, etc.."""
    return to_ms(timedelta(**kwargs))

def ms_scale(scale) -> float:
    """
    Determine a scale in milliseconds to work with.
    Accepts exact numbers or strings such as "hours", "minutes", etc..
    If given a string, we assume the scale is 1 of those (1 hour, 1 minute, etc)
    """

    if isinstance(scale, str):
        scale_to = ms(**{scale: 1})

    else:
        scale_to = scale

    return scale_to


class Win(tuple):
    """
    A fancy tuple object to define a single start-stop period with extra methods

    to generate a simple copy of this object you can do:

        win_ms = Win(*win_ms)

    """

    def __new__(cls, start, stop):
        """
        build a relative window in ms
        :param start: TimeDelta. If float, assuming it is in MILLISECONDS.
        :param stop: TimeDelta. If float, assuming it is in MILLISECONDS.
        :return: tuple object
        """
        start = to_ms(start)
        stop = to_ms(stop)

        # noinspection PyTypeChecker
        return super().__new__(cls, (start, stop))

    @classmethod
    def build_around(cls, ref, pre=0, post=0):
        """
        build a relative window in ms
        """
        # noinspection PyArgumentList
        return cls(ref, ref).extend(pre=pre, post=post)

    @classmethod
    def build_centered(cls, ref, duration):
        """
        build a relative window in ms
        """
        ref = to_ms(ref)
        duration = to_ms(duration)
        # noinspection PyArgumentList
        return cls(ref - duration * .5, ref + duration * .5)

    @property
    def start(self):
        """get the start time in milliseconds"""
        return self[0]

    @property
    def start_s(self):
        """get the start time in seconds"""
        return self.start * MS_TO_S

    @property
    def start_td(self):
        """get the start time as a timedelta object"""
        return timedelta(milliseconds=self.start)

    @property
    def stop(self):
        """get the stop time in milliseconds"""
        return self[1]

    @property
    def stop_s(self):
        """get the stop time in seconds"""
        return self.stop * MS_TO_S

    @property
    def stop_td(self):
        """get the stop time as a timedelta object"""
        return timedelta(milliseconds=self.stop)

    @property
    def length(self):
        """get the length in milliseconds"""
        return self.stop - self.start

    @property
    def length_s(self):
        """get the length in seconds"""
        return self.stop_s - self.start_s

    @property
    def length_td(self):
        """get the length in seconds"""
        return self.stop_td - self.start_td

    @property
    def mid(self):
        """get the mid point of the window"""
        return self.quantile_time(.5)

    def before(self, duration, offset=0):
        """
        Define a window immediately before this one with the given duration

        :param duration:
        :param offset:
        Pad the period between the new window and this one.
        If this number is negative, the windows will overlap.
        :return:
        """
        stop = self.start - offset
        return self.__class__(stop - duration, stop)

    def after(self, duration, offset=0):
        """
        Define a window immediately after this one with the given duration

        :param duration:
        :param offset:
        Pad the period between the new window and this one.
        If this number is negative, the windows will overlap.
        :return:
        """
        start = self.stop + offset
        return self.__class__(start, start + duration)

    def centered(self, duration, q='mid'):
        """build a new window centered in the middle of this one"""
        return self.__class__.build_centered(self.relative_time(q), duration)

    def arange(self, step):
        assert step > 0
        return np.arange(self.start, self.stop + step * .5, step)

    def contains(self, t):
        """check if a time stamp is contained in [start, stop]"""
        return (self.start <= t) & (t <= self.stop)

    def contained_in(self, win, fully=True) -> bool:
        """
        check if this window is contained in the given one

        :param win:

        :param fully: if True, window must be fully inside the given one to be considered contained.
            If False, a partial overlap is enough.

        :returns: a boolean series
        """
        win = Win(*win)

        if fully:
            start_within = win.contains(self.start)
            stop_within = win.contains(self.stop)

            return start_within & stop_within

        else:
            return self.overlaps(win)

    def overlaps(self, win) -> bool:
        """Check if this window and another one overlap"""
        start_late = win.stop <= self.start
        stop_early = self.stop <= win.start

        return (~start_late) & (~stop_early)

    def quantile_time(self, q: float) -> pd.Series:
        """
        Select a reference time as a quantile of the duration.

        :param q: float between 0 and 1. If 0 then time will be the 'start'. If 1, it will be 'stop'.
        :return:
        """
        return self.length * q + self.start

    def relative_time(self, q) -> float:
        """
        Get a time that is expressed relative to the window.
        :param q:
            If float, must be between 0 and 1. See quantile_time.
            If str, must be start, stop, mid, or ref.
        :return:
        """
        if isinstance(q, str):
            if q == 'mid':
                # noinspection PyTypeChecker
                return self.mid
            elif q == 'start':
                return self.start
            elif q == 'zero':
                return 0
            else:
                assert q == 'stop', f'Unknown time "{q}"'
                return self.stop

        else:
            # noinspection PyTypeChecker
            return self.quantile_time(q)

    def to_relative_time(self, time: float) -> float:
        """
        Get a time in relative time

        :param time:
        :return: float between 0 and 1. If 0 then time will be the 'start'. If 1, it will be 'stop'.
        """
        return (time - self.start) / self.length

    def assert_positive(self):
        """check stop comes after start"""
        assert self.length >= 0

    def to_str(self, plus_sign=False, strip=True, show_days=True):
        """pretty str format"""
        return (
            f'({strfdelta(self.start_td, plus_sign=plus_sign, strip=strip, show_days=show_days)},'
            f' {strfdelta(self.stop_td, plus_sign=plus_sign, strip=strip, show_days=show_days)})'
        )

    def __str__(self):
        """pretty str format"""
        return self.to_str()

    def _repr_html_(self):
        """pretty str format"""
        return self.__str__()

    def extend(self, pre=0., post=0.):
        """
        Add (or remove) time at the start (pre) or the end (post) of the window
        :param pre: time in milliseconds relative to start
        :param post: time in milliseconds relative to stop
        :return: a new tuple object
        """
        return Win(self.start + to_ms(pre), self.stop + to_ms(post))

    def subtract(self, exclusion_win):
        """
        Exclude a window of time from the current window.
        The result is 2 windows, potentially empty, before and after the exclusion_win
        :param exclusion_win:
        :return: a pair of Win corresponding to valid time before and after the exclusion win
        """
        pre = Win(
            min(self.start, exclusion_win.start),
            min(self.stop, exclusion_win.start),
        )

        post = Win(
            max(self.start, exclusion_win.stop),
            max(self.stop, exclusion_win.stop),
        )

        return pre, post

    def fragment(self, length_ms):
        """
        Break this window into multiple smaller ones that fit within.
        """
        breaks = self.arange(length_ms)
        return Windows.build_between(breaks)

    def shift(self, by=0.):
        """
        Shift the window without changing the duration
        :return: a new tuple object
        """
        return self.extend(pre=by, post=by)

    def shift_to_fit(self, other):
        """
        Shift this window so it fits within another window.
        This doesn't change the window duration.
        If 'other' is not big enough, a ValueError is raised.
        If this window is already within other, nothing is changed.

        This is useful when we want to load an example trace
        while making sure that the load window is within valid recording:

            load_win = load_win.shift_to_fit(raw.win_ms)

        :param other:
        :return:
        """
        left_shift = other[0] - self.start
        right_shift = other[1] - self.stop

        if left_shift > 0 > right_shift:
            raise ValueError(f'Window {self} can not be fit in {other}')

        elif left_shift > 0:
            return self.shift(left_shift)

        elif right_shift < 0:
            return self.shift(right_shift)

        return self

    def take_centered(self, max_duration):
        """
        Take a window within this window,
        centered in its middle and up to the given duration.

        This is useful when we want to load a reference trace
        around the middle of the experiment but we have experiments
        of various durations.
        """
        new = self.build_centered(ref=self.mid, duration=max_duration)
        return new.clip(self)

    def round(self, decimals=0, start=True, stop=True, scale='milliseconds'):
        """round this window """
        scale_to = ms_scale(scale)

        return self.__class__(
            np.round(self.start / scale_to, decimals=decimals) * scale_to if start else self.start,
            np.round(self.stop / scale_to, decimals=decimals) * scale_to if stop else self.stop,
        )

    def floor(self, start=True, stop=True, scale='milliseconds'):
        """round down to the closest integer for the given scale"""
        scale_to = ms_scale(scale)


        return self.__class__(
            np.floor(self.start / scale_to) * scale_to if start else self.start,
            np.floor(self.stop / scale_to) * scale_to if stop else self.stop,
        )

    def ceil(self, start=True, stop=True, scale='milliseconds'):
        """round down to the closest integer for the given scale"""
        scale_to = ms_scale(scale)


        return self.__class__(
            np.ceil(self.start / scale_to) * scale_to if start else self.start,
            np.ceil(self.stop / scale_to) * scale_to if stop else self.stop,
        )

    def floor_ceil(self, scale='milliseconds'):
        """
        Round start down and stop up.
        Useful to get a round window that for sure includes this one.
        """
        w = self
        w = w.floor(scale=scale, start=True, stop=False)
        w = w.ceil(scale=scale, start=False, stop=True)
        return w

    def ceil_floor(self, scale='milliseconds'):
        """
        Round start up and stop down.
        Useful to get a round window that for sure is included in this one (for example has valid data).
        """
        w = self
        w = w.ceil(scale=scale, start=True, stop=False)
        w = w.floor(scale=scale, start=False, stop=True)
        return w

    def round_loose(self, scale='milliseconds'):
        """Round start down and stop up, making the window looser"""
        return self.floor_ceil(scale)

    def round_tight(self, scale='milliseconds'):
        """Round start up and stop down, making the window tighter"""
        return self.ceil_floor(scale)


    def clip(self, other):
        """
        Clip this window so it fits within another window.
        If this window is already within other, nothing is changed.
        :param other:
        :return:
        """
        other = Win(*other)
        new = self.__class__(
            min(max(self.start, other.start), other.stop),
            max(min(self.stop, other.stop), other.start),
        )

        assert self.start <= self.stop
        assert self.length >= 0

        return new

    def to_slice_idx(self, stored_hz, load_hz=None):
        """
        Convert this window (in milliseconds) to a slice object of indices
        assuming a given sampling rate

        :param stored_hz: sampling rate of the stored data (typically 30kHz).

        :param load_hz: sampling rate that we want to load.
        This should be a divisor of the stored rate (defaults to the same).

        :return: slice object
        """
        if load_hz is None:
            load_hz = stored_hz

        assert_stride(stored_hz, load_hz, 'stored_hz', 'load_hz')

        from_i = int(np.round(stored_hz * self.start_s))
        to_i = int(np.round(stored_hz * self.stop_s))
        stride_i = get_stride(stored_hz, load_hz)

        return slice(from_i, to_i, stride_i)

    def to_slice_ms(self, step=None):
        """
        Convert this window to a slice object
        in milliseconds.

        This is useful to cut a pd.Series where the index is time:
            series.loc[win.to_slice_ms()]

        :return: slice object
        """
        return slice(self.start, self.stop, step)

    def crop_df(self, df: pd.DataFrame, reset=None, by=None) -> pd.DataFrame:
        """
        :param df:
        :param reset:
            Time to shift the df time window.
            It can be 'start' or 'stop'.
            Ff a float, it will be understood as a relative_time
        :param by:
        :return:
        """

        if by is None:
            time = df.index
        else:
            time = df[by].values

        mask = self.contains(time)
        df = df[mask].copy()

        if isinstance(reset, bool):
            reset = None if not reset else 'start'

        if reset is not None:
            reset = self.relative_time(reset)

            if by is None:
                df.index = df.index - reset
            else:
                df[by] = df[by] - reset

        return df

    def crop_ts(self, ts: np.ndarray, reset=None) -> np.array:
        """select from ts the times contained in this window"""
        new_ts = ts[self.contains(ts)]

        if isinstance(reset, bool):
            reset = None if not reset else 'start'

        if reset is not None:
            reset = self.relative_time(reset)
            new_ts = new_ts - reset

        return new_ts

    def interp_series(
            self,
            series: pd.Series,
            step,
            reset=None, shift=0,
            bounds_error=False,
    ) -> pd.Series:
        """
        Resample the data within the given window using linear interpolation.
        The result is similar to crop_df but the values are the result
        of an interpolation and the sampling rate of the result will be determined by this window,
        instead of being predetermined by the data original sampling.
        """

        assert np.all(series.index[:-1] < series.index[1:]), f'Index should be monotonic. Got: {series.index}'

        # for very long series, we can just cut it before doing any checks
        # noinspection PyTypeChecker
        idx_start: int = np.searchsorted(series.index, self.start) - 1
        idx_start = max(idx_start, 0)

        # noinspection PyTypeChecker
        idx_stop: int = np.searchsorted(series.index, self.stop) + 1
        idx_stop = min(idx_stop, len(series))

        series = series.iloc[idx_start:idx_stop]

        series = series.dropna()
        new_time = self.arange(step)

        if len(series) == 0:
            new_series = np.nan

        else:

            old_time = series.index.values - shift

            new_series = np.interp(
                new_time,
                old_time,
                series.values,
                left=np.nan if not bounds_error else None,
                right=np.nan if not bounds_error else None
            )

        if isinstance(reset, bool):
            reset = None if not reset else 'start'

        if reset is not None:
            reset = self.relative_time(reset)
            new_time = new_time - reset

        return pd.Series(new_series, index=new_time)

    def interp_df(self, df: pd.DataFrame, step, pbar=None, **kwargs) -> pd.DataFrame:
        """
        Resample the data within the given window using linear interpolation.

        Note: need to do series by series because scipy.interp gets confused if a matrix
        contains nans
        """
        if isinstance(df, pd.Series):
            # noinspection PyTypeChecker
            return self.interp_series(
                df,
                step,
                **kwargs,
            )

        df_iter = _optional_pbar(df.items(), total=len(df.columns), desc='interp', pbar=pbar)

        new_df = {
            k: self.interp_series(series, step, **kwargs)
            for k, series in df_iter
        }

        return pd.DataFrame(new_df, columns=df.columns)


class Windows(DataFrameWrapper):
    """Define periods of time to classify events or cut traces"""

    def __init__(self, windows):
        """
        :param windows: a windows DataFrame
        """
        assert 'start' in windows.columns
        assert 'stop' in windows.columns

        if windows.index.name is None:
            windows = windows.rename_axis(index='win_idx')

        super().__init__(windows)

    @property
    def wins(self):
        return self.reg

    def store_hdf(self, path: str, key):
        """
        save these windows in an HDF5 file
        """
        self.reg.to_hdf(path, key)

    @classmethod
    def load_hdf(cls, path: str, key):
        """
        load windows from an HDF5 file
        """
        # noinspection PyTypeChecker
        wins: pd.DataFrame = pd.read_hdf(path, key)
        return cls(wins)

    def to_str(self, split='\n'):
        """
        Serialize to human-readable text format.
        This drops most of the properties of the windows.
        For proper saving look at store_hdf / load_hdf.
        """
        desc = []

        for _, start, stop, cat in self.reg[['start', 'stop', 'cat']].itertuples():
            start = milliseconds_to_timestamp(start)
            stop = milliseconds_to_timestamp(stop)

            desc.append(
                f'{cat}-{start}-{stop}',
            )

        return split.join(desc)

    @classmethod
    def from_str(cls, string):
        """
        Serialize from human-readable text format.
        For proper saving look at store_hdf / load_hdf.
        """
        try:
            wins = []

            for win_str in string.strip().replace(';', '\n').split('\n'):
                cat = win_str[:win_str.find(':')]
                start, stop = win_str[win_str.find(':') + 1:].split('-')
                start = timestamp_to_milliseconds(start.strip())
                stop = timestamp_to_milliseconds(stop.strip())
                ref = start

                wins.append((cat, start, stop, ref))

            df = pd.DataFrame(wins, columns=['cat', 'start', 'stop', 'ref'])

            return cls(df)

        except ValueError:
            raise ValueError(f'Expected format "description: Xd HH:MM:SS.MS - Xd HH:MM:SS.MS". Got: "{string}"')

    @classmethod
    def build_around_df(cls, df, win=(0., 0.), col='time'):

        new = cls.build_around(df[col], win)
        new = new.add_cols(df.drop([col], axis=1))

        return new

    @classmethod
    def build_around(cls, marks, win=(0., 0.)):
        """
        build a dataframe containing several time windows around a series of time points
        for each row entry there are the start/stop points and the reference
        (sometimes used to compute the relative time).

        :param marks: a series of time points around which the windows are built
        :param win: a bi-tuple of time deltas, example: (-50, 150.)
        :return: a df that looks like:
                start    stop     ref
            0  1250.0  1400.0  1300.0
            1  1550.0  1700.0  1600.0
            2  1850.0  2000.0  1900.0
            3  2150.0  2300.0  2200.0
            4  2450.0  2600.0  2500.0
        """
        win = Win(*win)
        win.assert_positive()

        if not isinstance(marks, pd.Series):
            marks = pd.Series(np.asarray(marks))

        all_windows = marks.values[:, np.newaxis] + np.array(win)

        df = pd.DataFrame(
            {
                'start': all_windows[:, 0],
                'stop': all_windows[:, 1],
                'ref': marks,
            },
            index=marks.index)

        if df.index.name is None:
            df.index.name = 'win_idx'

        df.name = 'windows'

        return cls(df)

    def around(self, win, q='mid', old=None):
        """build a new set of windows around a reference time of the current ones"""
        new_wins = self.__class__.build_around(
            self.relative_time(q),
            win,
        )

        cols = self.reg.loc[:, self.columns_extra]

        if old is not None:
            cols = pd.concat([
                cols,
                self[['start', 'stop']].add_prefix(f'{old}_'),
            ], axis=1)

        new_wins = new_wins.add_cols(cols)

        return new_wins

    def centered(self, duration, q='mid', old=None):
        """build a new set of windows centered in the middle of these ones"""
        return self.around(
            Win.build_centered(0, duration),
            q=q,
            old=old,
        )

    def before(self, duration, offset=0, q='start', old=None):
        """build a new set of windows before the current ones"""
        return self.around(
            (-duration + offset, offset),
            q=q,
            old=old,
        )

    def after(self, duration, offset=0, q='stop', old=None):
        """build a new set of windows after the current ones"""
        return self.around(
            (offset, duration + offset),
            q=q,
            old=old,
        )

    def before_after(
            self,
            pre, post,
            pre_q='start', post_q='stop',
            pre_name='pre', post_name='post',
            col='when',
    ):
        new_wins = {
            pre_name:
                self.around(pre, q=pre_q)
                if isinstance(pre, (tuple, Win))
                else self.before(pre, q=pre_q),

            post_name:
                self.around(post, q=post_q)
                if isinstance(post, (tuple, Win))
                else self.after(post, q=post_q),
        }
        return Windows.concat(new_wins, cycle_name=col)

    @classmethod
    def build_centered(cls, marks, duration):
        """
        build selveral windows of the same duration around a given time point
        :param marks: a series of time points round which the windows are built
        :param duration:
        :return:
        """
        return cls.build_around(
            marks,
            win=Win.build_centered(0, duration=duration),
        )

    @classmethod
    def build_around_multiple(cls, marks, **wins):
        """
        Build multiple types of windows at once relative to each event.

        You can use this to build pre/post windows around important events:

            build_around_multiple(stimulus_times, pre=(-half_trial, 0), post=(0, half_trial))

        :param marks: a series of time points before which the windows are built
        :param wins: a dictionary from string to tuples (start, stop). The strings are used as categories
        :return:
        """

        all_windows = []
        for name, win in wins.items():
            windows = cls.build_around(marks, win).reg
            windows['cat'] = name
            all_windows.append(windows)

        all_windows = pd.concat(all_windows, axis=0, ignore_index=True)
        all_windows.sort_values(['ref', 'start', 'stop'], inplace=True)
        all_windows.reset_index(drop=True, inplace=True)
        all_windows.index.name = 'win_idx'

        return cls(all_windows)

    @classmethod
    def from_arrays(cls, start: np.ndarray, stop: np.ndarray, ref=.5):
        """
        build windows from arrays of start and stop times

        :param start:
        :param stop:
        :param ref: either an array or a value in [0, 1] indicating the relative
        interpolation between start and stop (by default: take the middle of the window).
        :return:
        """
        start = np.asarray(start)
        stop = np.asarray(stop)

        if np.issubdtype(type(ref), np.number):
            assert 0. <= ref <= 1
            ref: np.ndarray = start + (stop - start) * ref

            if np.allclose(ref, np.round(ref)):
                ref = np.round(ref).astype(int)

        return cls(pd.DataFrame.from_dict({
            'start': start,
            'stop': stop,
            'ref': ref,
        }))

    @classmethod
    def build_from_dict(cls, wins: dict, columns=('start', 'stop')):
        """
        build windows from dict of start and stop times. Example:

        wins = timeslice.Windows.from_dict({
            'fit': Win(ms(minutes=-15), 0),
            'pre': Win(ms(minutes=-30), ms(minutes=-15)),
            'post': Win(0, ms(minutes=+15)),
        })
        :param wins: dict with cat as key and values being a tuple of properties
        :param columns: the properties, by default (start, stop)
        """

        df = pd.DataFrame.from_dict(
            wins, orient='index', columns=columns,
        )

        df.rename_axis(index='cat', inplace=True)
        df.reset_index(inplace=True)

        if 'ref' not in df.columns:
            df['ref'] = df['start']

        return cls(df)

    @classmethod
    def build_sliding(
            cls,
            start_ms,
            stop_ms,
            length_ms,
            sampling_rate,
            start_off_ms=0,
            stop_off_ms=0,
            **kwargs
    ):
        """
        Build equal-sized sliding windows in SAMPLES to analyze a time series.
        See build_sliding_samples

        :param start_ms: when the recording starts (assuming sample=0 equals this time)
        :param stop_ms: when the recording stops (assuming sample=-1 equals this time)
        :param sampling_rate: take a guess
        :param kwargs: other args to build_sliding_samples

        :param length_ms:
        :param start_off_ms:
        :param stop_off_ms:

        :return:
        """
        win_samples = cls.build_sliding_samples(
            start_ms=0 + start_off_ms,
            stop_ms=(stop_ms - start_ms) + stop_off_ms,
            sampling_rate=sampling_rate,
            length_ms=length_ms,
            **kwargs,
        )

        win_ms = win_samples.sample_to_ms(
            sampling_rate,
            tstart=start_ms,
        )

        win_samples = win_samples.add_cols(win_ms.reg[['start', 'stop', 'ref']].add_suffix('_ms'))

        return win_samples

    @classmethod
    def build_sliding_samples(
            cls,
            start_ms,
            stop_ms,
            length_ms,
            sampling_rate,
            step_ms=None,
            ignore_remaining=True,
    ):
        """
        Build equal-sized sliding windows in SAMPLES.

        Note that we specify the window properties in ms and the resulting windows
        will be as close as possible to those, but since they must live in
        sample space, they will not be exactly the same.
        Use sample_to_ms to recover the actual times

        :param ignore_remaining:
        :param start_ms: earliest time to start a window
        :param stop_ms: latest time to stop a window
        :param length_ms: length of each window
        :param step_ms: spacing between the center of the windows. By default 1 sample.
        :param sampling_rate: sampling rate of the signal that these windows will be applied to.
        :return:
        """
        start_ms = to_ms(start_ms)
        stop_ms = to_ms(stop_ms)
        length_ms = to_ms(length_ms)

        start_idx = int(np.round(start_ms * MS_TO_S * sampling_rate))
        stop_idx = int(np.round(stop_ms * MS_TO_S * sampling_rate))

        if step_ms is None:
            step_idx = 1
        else:
            step_ms = to_ms(step_ms)
            step_idx = max(1, int(np.round(step_ms * MS_TO_S * sampling_rate)))

        length_idx = max(1, int(np.round(length_ms * MS_TO_S * sampling_rate)))

        win_start_idcs = np.arange(start_idx, stop_idx, step_idx)
        win_stop_idcs = win_start_idcs + length_idx

        if ignore_remaining:
            mask = (start_idx <= win_start_idcs) & (win_stop_idcs <= stop_idx)
            win_start_idcs = win_start_idcs[mask]
            win_stop_idcs = win_stop_idcs[mask]
        else:
            win_start_idcs = np.clip(win_start_idcs, start_idx, stop_idx)
            win_stop_idcs = np.clip(win_stop_idcs, start_idx, stop_idx)

        wins = cls.from_arrays(
            start=win_start_idcs,
            stop=win_stop_idcs,
        )

        return wins

    @classmethod
    def build_between(cls, times: np.ndarray, start=None, stop=None):
        """
        build a dataframe of windows between every pair of given markers

        :param times: timepoints that define the windows
        This is expected as a pd.Series or as a DataFrame with a 'time' column.
        If a DataFrame, all extra columns will be appended in the resulting windows
        with a prefix start_ or  stop_.

        This means you can call it like:

            build_windows_between(all_sw[['time', 'sleep_cat']])

        and get:
                        start        stop start_sleep_cat stop_sleep_cat         ref  length
            0          1000.0      4000.0         unknown        unknown      1000.0  3000.0
            1          4000.0     10200.0         unknown        unknown      4000.0  6200.0
            2         16700.0     20400.0         unknown            sws     16700.0  3700.0
            ...           ...         ...             ...            ...         ...     ...
            13838  54715200.0  54718900.0             sws            sws  54715200.0  3700.0

        :param start: if present, add a window between "start" to the start of the first window.
        If present, the index of times will be reset.

        :param stop: if present, add a window between the stop of the last window to "stop"
        If present, the index of times will be reset.

        :return:
        """

        if isinstance(times, (np.ndarray, list, tuple)):
            times = pd.Series(times, name='time')

        if isinstance(times, pd.Series):
            times = times.rename('time').to_frame()

        assert isinstance(times, pd.DataFrame)
        assert 'time' in times.columns

        times = times.sort_values('time')

        if start is not None:
            endpoint = np.min(times['time'])
            if start < endpoint or np.isnan(endpoint):
                times = pd.concat([
                    pd.DataFrame({'time': [start]}),
                    times,
                ], axis=0, ignore_index=True)
            else:
                logging.warning(f'Cannot set start at {start} when windows begin earlier at {endpoint}')

        if stop is not None:
            endpoint = np.max(times['time'])
            if stop > endpoint or np.isnan(endpoint):
                times = pd.concat([
                    times,
                    pd.DataFrame({'time': [stop]}),
                ], axis=0, ignore_index=True)
            else:
                logging.warning(f'Cannot set stop at {stop} when windows begin later at {endpoint}')

        start = times.iloc[:-1]
        stop = times.iloc[1:]

        wins = pd.DataFrame({
            'start': start['time'].values,
            'stop': stop['time'].values,
        })

        wins['ref'] = wins['start']

        for prefix, ref in [('start_', start), ('stop_', stop)]:
            for name, values in ref.drop('time', axis=1).add_prefix(prefix).items():
                wins[name] = values.values

        return cls(wins)

    @classmethod
    def build_from_transitions(cls, values: pd.Series, sampling_period: float):
        """create a window for each change of value in "values" or time gap"""
        time_jump = ~np.isclose(np.diff(values.index), sampling_period)
        mode_jump = values.values[:-1] != values.values[1:]

        transition_idcs, = np.where(time_jump | mode_jump)

        all_transitions = []

        for i in tqdm(transition_idcs):
            t0, t1 = values.index[[i, i + 1]]

            all_transitions.append({
                'start': t0,
                'stop': t1,
                'start_state': values.loc[t0],
                'stop_state': values.loc[t1],
            })

        all_transitions = pd.DataFrame(all_transitions)
        all_transitions['mid_time'] = all_transitions[['start_time', 'stop_time']].mean(axis=1)

        assert np.all(all_transitions['stop_state'].values[:-1] == all_transitions['start_state'].values[1:])

        return cls(all_transitions)

    @classmethod
    def build_from_contiguous_values(cls, values, mid_sr=True, include_right=False):
        """
        detect contiguous regions of discrete values in a time series
        and keep the start/stop idcs

        :param values:
        :param mid_sr:
        :param include_right: whether windows are [start, stop) or [start, stop]

        If True, then consecutive windows will look like: [0, 10] [11, 20],...
        If False, then consecutive windows will look like: [0, 10) [10, 20),...

        Note that if the times for the values are float (e.g. milisecond), then windows
        that include_right are technically not tight, since there is time between 10 and 11.

        :return: DF

            example:

                   start    stop      cat
            0          0       1  unknown
            1          1      29      sws
            ...      ...     ...      ...
            3236  218808  218955  unknown
            3237  218955  218989      sws
        """
        if values.empty:
            return cls(pd.DataFrame({'start': [], 'stop': [], 'ref': [], 'cat': []}))

        assert pd.Series(np.asarray(values)).notna().all(), \
            'Can not extract unique values with nans. Please drop or replace.'

        unique_values, codes = np.unique(values, return_inverse=True)

        transitions = np.diff(codes) != 0

        # diff drops the first element
        transitions_idcs, = np.where(transitions)

        wins = pd.DataFrame.from_dict({
            'start': np.append(0, transitions_idcs),
            'stop': np.append(transitions_idcs, len(codes) - 1),
        })

        wins['cat'] = unique_values[codes[np.append(True, transitions)]]

        if include_right:
            wins['stop'] = wins['stop'] - 1

        assert np.all(wins['start'] <= wins['stop'])
        wins['ref'] = wins['start']

        if isinstance(values, pd.Series):
            wins = cls(wins).sample_to_ms_by_time_index(values.index).reg

        if mid_sr:
            sampling_periods: np.array = np.unique(np.diff(values.index))
            assert np.allclose(sampling_periods[0], sampling_periods), sampling_periods
            sampling_period = sampling_periods[0]
            wins[['start', 'stop', 'ref']] += sampling_period * .5
            wins.loc[wins['start'].idxmin(), 'start'] -= sampling_period * 1

        return cls(wins)

    @classmethod
    def concat(cls, many, cycle_name='cycle_idx', local_name=None, **kwargs):
        """
        """

        if hasattr(many, 'items'):
            dfs = {
                idx: c.reg for idx, c in many.items()
            }
        else:
            dfs = {
                idx: c.reg for idx, c in enumerate(many)
            }

        results = pd.concat(dfs, **kwargs)

        if local_name is None:
            old_index_name = results.index.names[1]
            if old_index_name == '':
                old_index_name = 'win_idx'

            local_name = f'original_{old_index_name}'

        if not isinstance(cycle_name, list):
            cycle_name = [cycle_name]
        results.rename_axis(index=cycle_name + [local_name], inplace=True)
        results.reset_index(inplace=True)
        results.rename_axis(index='win_idx', inplace=True)

        return cls(results)

    @classmethod
    def concat_list(cls, many, axis=0, reset_index=True, **kwargs):
        """
        """
        dfs = [c.reg for c in many]
        results = pd.concat(dfs, axis=axis, **kwargs)

        if reset_index:
            results.reset_index(inplace=True, drop=True)
            results.rename_axis(index='win_idx', inplace=True)

        return cls(results)

    def rename_index(self, name):
        return self.__class__(self.reg.rename_axis(index=name))

    def generate_cat(self, times: pd.Index, col='cat', right=False) -> pd.Series:
        """
        Take the category for each given time point.
        This generates a result similar to "generate_cat_contiguous", but it takes
        an explicit index instead of a sampling rate.
        This is a short hand for classify_events.

        Example:
            rem_index = rem_wins.generate_cat(all_beta.index)

        :param times:
        :param col:
        :param right:
        :return:
        """
        assert pd.Index(times).is_unique

        classified = self.classify_events(times, right=right)
        classified = classified.set_index(times[classified.index])

        return classified[col].reindex(times)

    def generate_cat_contiguous(
            self, sampling_period, start=None, stop=None,
            dim='cat', pbar=None,
            fillna=False,
    ) -> pd.Series:
        """
        Generate a series containing the categories indicated by these windows

        Reverse to build_from_contiguous_values

        :param sampling_period:
        :param pbar:
        :param fillna:
        :param dim:
        :param start: by default, start of earliest window
        :param stop: by default, stop of latest window
        :return: a series with time as index and 'cat' as values
        """
        assert self.are_exclusive()

        if start is None:
            start = self.reg[['start', 'stop']].values.min()

        if stop is None:
            stop = self.reg[['start', 'stop']].values.max()

        index = np.arange(start, stop, sampling_period)

        mode = pd.Series(fillna, index=index)

        slicing = self.reg[['start', 'stop', dim]].itertuples()

        slicing = _optional_pbar(slicing, total=len(self), desc='gen values', pbar=pbar)

        for _, start, stop, cat in slicing:
            mode.loc[start:stop] = cat

        return mode

    def reset_index(self, sort_by=('start', 'stop', 'ref'), drop=True):
        """return a copy of these windows where the windows have been sorted in time and the index reset"""
        sort_by = pd.Index(list(sort_by)).intersection(self.reg.columns)
        wins = self.reg.sort_values(list(sort_by)).reset_index(drop=drop)
        return self.__class__(wins)

    def scale(self, factor, cols=('start', 'stop', 'ref')):
        cols = list(cols)

        new = self.copy()
        new.reg[cols] *= factor
        return new

    def ms_to_sample(self, sampling_rate):
        """
        Convert windows from "time" (float, in milliseconds) to "sample" (integer).
        """
        # TODO what to do when windows are in relative time and sampling rate is low?
        # The best value to round can be chosen only for absolute time.
        assert not self.are_in_samples()

        new = self.copy()
        cols = ['start', 'stop', 'ref']
        new.reg[cols] = np.round(sampling_rate * new.reg[cols].values * MS_TO_S).astype(int)

        assert new.are_in_samples()

        return new

    def sample_to_ms(self, sampling_rate, tstart=0):
        """
        Convert windows from "sample" (integer) to "time" (float, in milliseconds).
        """
        wins_ms = self.copy()

        time_cols = ['start', 'stop', 'ref']
        wins_ms.reg[time_cols] = S_TO_MS * self.reg[time_cols] / sampling_rate + to_ms(tstart)

        if 'length' in wins_ms.reg.columns:
            wins_ms.reg['length'] = wins_ms.lengths()

        return wins_ms

    def sample_to_ms_by_time_index(self, index):
        """
        Converts these windows, which must be in sample space (integers),
        to times using the given time index.
        The index must contain all corresponding to these windows,
        or it must be regular so that a sampling rate can be estimated.
        Index is assumed in milliseconds
        """
        tstep = np.diff(index)
        if np.allclose(tstep, tstep[0]):
            sampling_rate = 1. / (tstep[0] * MS_TO_S)
            return self.sample_to_ms(sampling_rate, tstart=np.min(index))

        else:
            wins = self.reg.copy()

            for col in 'start', 'stop', 'ref':
                if col in wins.columns:
                    wins[col] = index[wins[col]]

            return self.__class__(wins)

    def rename_cat(self, renaming, *, col='cat', **renaming_kwargs):
        """
        Change one or more category names

        Useful for example to decide that Voltage Clamp experiments should be considered baseline.
        Use like:
            new_wins = wins.rename_cat(VC='baseline')

        or like:
            Windows.build_from_contiguous_values(beta > .1).rename_cat({False: 'sws', True: 'rem'})

        :param col:
        :param renaming:
        :return:
        """
        renaming = {**renaming, **renaming_kwargs}

        wins = self.copy()

        wins.reg[col] = wins.reg[col].map(
            lambda x: renaming[x] if x in renaming else x
        )

        return wins

    def complement(self, cat='unknown', reset_index=False, start=None, stop=None):
        """make windows cover all of the time by adding new ones for the periods missing (see invert)"""

        inv = self.invert(start=start, stop=stop)

        wins_ms = pd.concat(
            [self.reg, inv.reg],
            axis=0, ignore_index=True, sort=False)

        wins_ms = Windows(wins_ms)

        if 'cat' in wins_ms.reg.columns:
            wins_ms.reg['cat'].fillna(cat, inplace=True)

        if 'length' in wins_ms.reg.columns:
            wins_ms.reg['length'] = wins_ms.lengths()

        wins_ms.reg.sort_values('start', inplace=True)

        if reset_index:
            wins_ms.reg.reset_index(drop=True, inplace=True)

        return wins_ms

    def iter_wins(self, pbar=None):
        for idx, ref, win in self.iter_wins_ref(pbar=pbar):
            yield idx, win

    def iter_wins_ref(self, pbar=None):
        for idx, win, props in self.iter_wins_items(pbar=pbar):
            yield idx, props['ref'], win

    def iter_wins_items(self, pbar=None):
        """
        Iterate over the windows with all of their properties.
        :returns: Iterable where the returned items are tuples:
            <window_index, win, properties>
        where win is of type timeslice.Win and properties is a pd.Series

        Use like:
            for idx, win, props in wins.iter_wins_items():
                print(win, props['cat'])
        """
        other_cols = self.reg.columns.difference(['start', 'stop'])

        it = self.reg.T.items()

        it = _optional_pbar(it, total=len(self.reg), pbar=pbar)

        for idx, props in it:
            yield idx, Win(props['start'], props['stop']), props[other_cols]

    def iter_groupby(self, *args, pbar=None, **kwargs):
        """Iterate these windows after pd.groupby. Objects returned will be of Windows type."""
        grouped = self.groupby(*args, **kwargs)

        grouped = _optional_pbar(grouped, total=len(grouped), pbar=pbar)

        for key, group in grouped:
            yield key, Windows(group)

    def get(self, win_idx=None) -> Win:
        """return a single window. If no index it's given, we assume there is only one"""
        if win_idx is None:
            assert len(self.reg.index) == 1
            win_idx = self.index[0]

        start, stop = self.reg.loc[win_idx, ['start', 'stop']]
        return Win(start, stop)

    def get_rel(self, win_idx=None) -> Win:
        """
        return a single window relative to its reference.
        If no index it's given, we assume there is only one.
        """
        if win_idx is None:
            assert len(self.reg.index) == 1
            win_idx = self.index[0]

        return self.get(win_idx).shift(-self.loc[win_idx, 'ref'])

    def get_props(self, win_idx=None) -> pd.Series:
        """
        return the properties of a single window.
        If no index it's given, we assume there is only one.
        """
        if win_idx is None:
            assert len(self.reg.index) == 1
            win_idx = self.index[0]

        return self.reg.loc[win_idx]

    def crop_df(self, df: pd.DataFrame, reset='ref', by=None, pbar=None) -> dict:

        sections = {}

        for idx, win in self.iter_wins(pbar=pbar):
            win: Win

            # look up reset time specific for this window
            win_reset = reset
            if isinstance(reset, str):
                win_reset = win.to_relative_time(self.reg.loc[idx, reset])

            sections[idx] = win.crop_df(df, by=by, reset=win_reset)

        return sections

    def interp_series(self, s: pd.Series, step: float, reset='ref', pbar=None):

        sections = {}

        for idx, win in self.iter_wins(pbar=pbar):
            win: Win

            # look up reset time specific for this window
            win_reset = reset
            if isinstance(reset, str):
                win_reset = win.to_relative_time(self.reg.loc[idx, reset])

            sections[idx] = win.interp_series(s, step=step, reset=win_reset)

        return sections

    def interp_df(self, df: pd.DataFrame, step: float, reset='ref', pbar=None):

        sections = {}

        for idx, win in self.iter_wins(pbar=pbar):
            win: Win

            # look up reset time specific for this window
            win_reset = reset
            if isinstance(reset, str):
                win_reset = win.to_relative_time(self.reg.loc[idx, reset])

            sections[idx] = win.interp_df(df, step=step, reset=win_reset, pbar=False)

        return sections

    # Methods from pandas DataFrame
    def copy(self):
        return self.__class__(self.reg.copy())

    @functools.wraps(pd.DataFrame.groupby)
    def groupby(self, *args, **kwargs):
        return self.reg.groupby(*args, **kwargs)

    @functools.wraps(pd.DataFrame.drop)
    def drop(self, *args, **kwargs):
        return self.__class__(self.reg.drop(*args, **kwargs))

    @property
    def columns_extra(self):
        """return all columns EXCEPT the time windows"""
        return self.reg.columns.drop(['start', 'stop', 'ref'])

    @property
    def loc(self):
        """pd.DataFrame accessor"""
        return self.reg.loc

    @property
    def iloc(self):
        """pd.DataFrame accessor"""
        return self.reg.iloc

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.reg._repr_html_()

    def describe(self, quiet=False):
        """
        human-readable description of the windows
        :param quiet: if True the description is returned otherwise it is printed
        :return:
        """
        desc_tight = 'tight ' if self.are_tight() else ''
        desc_exclusive = 'exclusive ' if self.are_exclusive() else 'non-exclusive '
        desc_uniform = 'uniform ' if self.are_uniform() else ''

        desc = (
            f'{len(self):,g} {desc_tight}{desc_uniform}{desc_exclusive}wins'
            f' covering {strf_ms(self.total())}'
        )

        if 'cat' in self.reg.columns:
            desc_cats = ', '.join([f'{strf_ms(v)} {k}' for k, v in self.total_by_cat().items()])
            desc = f'{desc} ({desc_cats})'

        if quiet:
            return desc
        else:
            print(desc)
            return None

    def are_uniform(self, atol=1.e-8) -> bool:
        """
        :return:
            wheter all windows are instantiation of the same template Win
            just with different ref times
        """
        if len(self) == 0:
            return True

        pre = (self.reg['start'] - self.reg['ref'])
        post = (self.reg['stop'] - self.reg['ref'])

        uni_pre = np.allclose(pre.values, pre.values[0], atol=atol)
        uni_post = np.allclose(post.values, post.values[0], atol=atol)

        return uni_pre and uni_post

    def are_alternating(self, col='cat'):
        """
        :return:
            wheter window categories are alternating between two values
        """
        this_win = self[col]
        if this_win.isna().any():
            return False

        prev_win = this_win.shift(1)
        if not np.all(this_win != prev_win):
            return False

        prev_prev_win = prev_win.shift(1)
        if not np.all(this_win.iloc[2:] == prev_prev_win.iloc[2:]):
            return False

        return True

    def force_uniform(self, length=None):
        """
        Ensure all windows are uniform by adjusting those too big or small.
        Ref times are left intact.
        Start and stop are shifted proportionately to what they were before
        so all windows have the same length.
        :return:
        """

        lengths = self.lengths()

        if length is None:
            length = lengths.median()

        ref = self.relative_time('ref')

        before = self.reg['start'] - ref
        after = self.reg['stop'] - ref

        prop = length / lengths

        result = self.copy()

        result.reg['start'] = prop * before + ref
        result.reg['stop'] = prop * after + ref

        assert result.are_uniform(), result.lengths()

        return result

    def get_edges(self):
        return np.sort(np.unique(self.reg[['start', 'stop']].values.flatten()))

    def get_breaks(self):
        assert self.are_tight()
        return np.append(self['start'].values, self['stop'].values[-1])

    def are_exclusive(self) -> bool:
        """
        :return: whether the windows monotonically increase
        """
        if len(self) == 1:
            return True

        edges = self.reg[['start', 'stop']].sort_values(['start', 'stop']).values.flatten()

        # noinspection PyTypeChecker
        return np.all(np.diff(edges) >= 0)

    def are_integer(self) -> bool:
        """
        :return: whether the windows are expressed in integer values. Usually to represent sample indices.
        """
        return (
                np.issubdtype(self.reg['start'].dtype, np.integer) and
                np.issubdtype(self.reg['stop'].dtype, np.integer)
        )

    def are_tight(self, inclusive=None) -> bool:
        """
        :param: whether the start/stop edges are inclusive (only for integer wins).
        In that case, there may be a gap of "1" between each two windows
        :return: whether the windows leave gaps between them
        """
        if len(self) == 1:
            return True

        if inclusive is None and self.are_integer():
            inclusive = True

        if inclusive:
            assert self.are_integer(), 'inclusive windows should be defined with integer edges'

        edges = self.reg[['start', 'stop']].sort_values(['start', 'stop'])
        return (edges['start'].values[1:] - edges['stop'][:-1]).max() <= (1 if inclusive else 0)

    def are_in_samples(self) -> bool:
        """check if current windows are defined in "samples" (integer) rather than "time" (float) """

        # noinspection PyTypeChecker
        return np.all([np.issubdtype(t, np.integer) for t in self.reg.dtypes[['start', 'stop', 'ref']]])

    def is_ref_inside(self) -> pd.Series:
        """check if ref is between start and stop"""
        # noinspection PyTypeChecker
        return (self['start'] <= self['ref']) & (self['ref'] <= self['stop'])

    def is_within(self, win: Win) -> pd.Series:
        """check if each window fits within a bigger one (partial overlaps give False)"""
        win = Win(*win)
        # noinspection PyTypeChecker
        return (win.start <= self['start']) & (self['stop'] <= win.stop)

    def is_within_any(self, other) -> pd.Series:
        """
        Check if each window fits within any of the given ones (partial overlaps give False).
        Useful if you have a set of "valid" periods and you just want to select the windows that fully in them.
        """
        valid = np.zeros(len(self), dtype=bool)
        for _, win in other.iter_wins():
            valid = valid | self.is_within(win)

        return valid

    def is_empty(self):
        return self['stop'] <= self['start']

    def crop_to_main(self, win_ms, reset=False):
        """
        Crop windows to only contain time inside win_ms.
        Windows outside will be dropped, windows at the edge will be cut and windows inside will be left intact.

        :param win_ms:
        :param reset: if true, the returned windows start as if win_ms.start was time=0
        :return:
        """
        win_ms = Win(*win_ms)

        new = self.copy()

        new.reg['start'] = np.maximum(new.reg['start'], win_ms.start)
        new.reg['stop'] = np.minimum(new.reg['stop'], win_ms.stop)

        new.reg = new.reg[new.reg['start'] <= new.reg['stop']]

        if reset:
            new = new.shift(-win_ms.start)

        return new

    def crop_to_multiple(self, others, pbar=None, drop=False):
        """
        Crop all of these windows so they only cover periods covered in 'others'.
        Note that this may drop original windows, reduce their size or even
        fragment them into multiple smaller ones.
        Original index is kept as a column "original_win_idx".

        :param others: another Windows instance
        :param pbar:
        :param drop: whether to keep the old win_idx as a column "original_win_idx"
        :return:
        """
        assert self.are_exclusive() and others.are_exclusive()

        others_bounds = others.reg[['start', 'stop']].itertuples()

        others_bounds = _optional_pbar(others_bounds, total=len(others), pbar=pbar)

        all_cropped = []
        for _, start, stop in others_bounds:
            cropped = self.crop_to_main((start, stop))

            cropped = cropped.reg.rename_axis(index='original_win_idx').reset_index(drop=drop)

            all_cropped.append(cropped)

        all_cropped = pd.concat(all_cropped, axis=0, ignore_index=True)
        all_cropped = Windows(all_cropped)

        return all_cropped

    def crop_to_minimum_common(self):
        """
        Given a bunch of windows in "sample" space,
        Crop some of them so they all have the same length before and after "ref".

        This is useful to avoid rounding errors when converting between time and sample space
        that produce many traces of 900 samples and the odd few of 901 or 899

        both input and output looks like:

                        start       stop        ref
            1821       905793     906693     906093
            3330      1721074    1721974    1721374
            ...           ...        ...        ...
            243023  128684232  128685132  128684532

        """
        assert self.are_in_samples()

        new = self.copy()

        win = (
            (new.reg['ref'] - new.reg['start']).min(),
            (new.reg['stop'] - new.reg['ref']).min(),
        )

        new.reg['start'] = new.reg['ref'] - win[0]
        new.reg['stop'] = new.reg['ref'] + win[1]

        assert len(new.lengths().drop_duplicates()) <= 1

        return new

    def __len__(self):
        """number of windows"""
        return len(self.reg)

    def lengths(self) -> pd.Series:
        """length of each window"""
        return self.reg['stop'] - self.reg['start']

    def mid(self) -> pd.Series:
        """middle time of each window"""
        return self.quantile_time(0.5)

    def contain(self, t, how='any'):
        """
        check if any (or all) of these windows contain t
        :param t: single number, np.ndarray or pd.Series
        :param how:
        """
        does_it = np.array([w.contains(t) for idx, w in self.iter_wins()])

        if how == 'any':
            res = np.any(does_it, axis=0)
        else:
            assert how == 'all'
            res = np.all(does_it, axis=0)

        if isinstance(t, pd.Series):
            return pd.Series(res, index=t.index)

        return res

    # noinspection PyTypeChecker
    def contained_in(self, win, fully=True) -> pd.Series:
        """
        check, for each window, if its contained in the given one

        :param win:
        :param fully: if True, only windows that are fully contained will return True.
            If False, windows that partially overlap win will also return True.

        :returns: a boolean series
        """

        win = Win(*win)

        if fully:
            start_within = np.array([win.contains(w.start) for idx, w in self.iter_wins()])
            stop_within = np.array([win.contains(w.stop) for idx, w in self.iter_wins()])

            return start_within & stop_within

        else:
            start_late = np.array([win.stop <= w.start for idx, w in self.iter_wins()])
            stop_early = np.array([w.stop <= win.start for idx, w in self.iter_wins()])

            return (~start_late) & (~stop_early)

    def quantile_time(self, q: float) -> pd.Series:
        """
        Select a reference time for each window as a quantile of the duration.

        :param q: float between 0 and 1. If 0 then time will be the 'start' of each window. If 1, it will be 'stop'.
        :return:
        """
        # noinspection PyTypeChecker
        return self.lengths() * q + self.reg['start']

    def relative_time(self, q) -> pd.Series:
        """
        Get a time for each window that is expressed relative to the window.
        :param q:
            If float, must be between 0 and 1. See quantile_time.
            If str, must be start, stop, mid, or ref.
        :return:
        """
        if isinstance(q, str):
            if q == 'mid':
                return self.mid()
            else:
                return self.reg[q]
        else:
            return self.quantile_time(q)

    def reset_ref(self, q):
        """
        Reset the ref for all windows.
        :param q: see relative_time.
        """
        new = self.copy()
        new['ref'] = new.relative_time(q)
        return new

    def total(self) -> float:
        """total time covered by these windows"""
        return self.lengths().sum()

    def total_by_cat(self, by='cat'):
        """return a series matching each category to the total time covered by its windows"""

        if isinstance(by, str):
            by = self.reg[by]

        return (self.reg['stop'] - self.reg['start']).groupby(by).sum()

    def get_rel_win(self, ref='ref', atol=1.e-8, ):
        """
        Get the original window (pre, post) relative to ref
        windows must all have the same rel_window within some tolerance.

        Average window edges is taken (only relevant for very high tolerance).
        :return:
        """
        assert self.are_uniform(atol=atol)

        ref_t = self.relative_time(ref)

        pre = (self.reg['start'] - ref_t)
        post = (self.reg['stop'] - ref_t)

        return Win(pre.mean(), post.mean())

    def get_global_win(self):
        """
        Get the minimum window that includes all of these windows
        :return:
        """
        start = self.reg[['start', 'stop']].values.min()
        stop = self.reg[['start', 'stop']].values.max()
        return Win(start, stop)

    def add_cols(self, extra: pd.DataFrame):
        """add extra columns describing properties of these windows"""
        assert len(extra) == len(self)
        return self.__class__(pd.concat([self.reg, extra], axis=1))

    def merge_cols(self, extra: pd.DataFrame, **kwargs):
        """merge in extra columns describing properties of these windows"""
        return self.__class__(pd.merge(self.reg, extra, **kwargs))

    def set_cols(self, extra: pd.DataFrame, *, suffix=None, prefix=None):
        """add extra columns describing properties of these windows"""
        assert len(extra) == len(self)
        new = self.copy()

        if suffix is not None:
            extra = extra.add_suffix(suffix)

        if prefix is not None:
            extra = extra.add_prefix(prefix)

        for col, vals in extra.items():
            new[col] = vals
        return new

    def sel_length_between(self, vmin=-np.inf, vmax=+np.inf):
        return self.sel_mask(self.lengths().between(vmin, vmax))

    def fragment(self, length_ms, align='left'):
        """
        Break these windows into smaller ones that fit within.
        This is useful if windows are uneven but we want to compute something evenly.
        Note that you may loose coverage if length_ms is not a perfect multiple
        of ALL of the windows.
        Lost coverage is distributed according to "align".

        :param length_ms:
        :param align: either 'left', 'right', or a float between 0 and 1 indicating
        how lost time should be allocated:
            left (or 0) fragments are aligned to the left and lost time is on the right
            right (or 1) opposite.

        :return:
        """

        length_ms = to_ms(length_ms)

        if isinstance(align, str):
            assert align in ['left', 'right']
            align = dict(left=0., right=1.)[align]

        assert align >= 0
        assert align <= 1.

        all_fragments = []

        for idx in self.reg.index:

            rel_win = self.reg.loc[idx, ['start', 'stop']].values

            edges = np.arange(*rel_win, length_ms)

            if len(edges) > 1:
                offset = (rel_win[1] - np.max(edges)) * align

                edges = edges + offset

                all_fragments.append(Windows.build_between(edges).reg)
                all_fragments[-1]['main_win_idx'] = idx

        if len(all_fragments) > 0:
            return self.__class__(pd.concat(all_fragments, ignore_index=True))
        else:
            return self.__class__(pd.DataFrame(columns=['start', 'stop', 'ref']))

    def shift(self, shifts, dropmissing=True):
        """apply a shift to every window"""

        if np.issubdtype(type(shifts), np.number):
            shifts = np.ones(len(self)) * shifts

        if not isinstance(shifts, pd.Series):
            shifts = pd.Series(np.asarray(shifts), index=self.reg.index)

        shifts = shifts.reindex(self.reg.index)

        new = self.reg.copy()

        if dropmissing:
            shifts = shifts.dropna()
            new = new.reindex(shifts.index)

        for c in 'start', 'stop', 'ref':
            new[c] = new[c] + shifts

        # note that we can no longer expect them to be exclusive
        return Windows(new)

    def defrag(self, start=0):
        """
        If these windows are not tight, shift them so they all follow one another
        This preserves any metadata, the relative window order and the window index.
        This is useful to remove invalid sections of an experiment and just patch together the valid ones.

        See defrag_events and defrag_series
        """

        last = start
        cols = ['start', 'stop', 'ref']
        shifted_wins = {}

        for win_idx in self.reg.index:
            shifted_wins[win_idx] = self.reg.loc[win_idx].copy()

            shifted_wins[win_idx][cols] = shifted_wins[win_idx][cols] - shifted_wins[win_idx]['start'] + last

            last = shifted_wins[win_idx]['stop']

        df = pd.DataFrame.from_dict(shifted_wins, orient='index').rename_axis(index=self.reg.index.name)
        shifted_wins = self.__class__(df)

        assert shifted_wins.are_tight()
        assert shifted_wins.are_exclusive()

        return shifted_wins

    def defrag_events(self, times: pd.Series, push_inbetween=False):
        """
        Take the given times and return them after shifting them so that all of the windows
        are consecutive.

        see defrag and defrag_series

        example:

            new_times = sleep_wins.sel(cat='rem').defrag_events(spks.spikes['time'])

        :param times:
            multiple time stamps
            Index is some sort of name, value is time in milliseconds.

        :param push_inbetween:
            Policy for items that fall between windows.
            By default, they are dropped.
            If True, they are pushed to the new shifted edge of the surrounding windows.
        """
        shifted_wins = self.defrag()

        classified_events = self.classify_events(times)

        offsets = shifted_wins.reg['start'].reindex(classified_events['win_idx']).values

        shifted_times = classified_events['delay'] + offsets

        if push_inbetween:
            inverted_wins = self.invert()

            stop_to_win_idx = self.reg.reset_index().set_index('stop')['win_idx']
            inverted_win_prev = stop_to_win_idx.reindex(inverted_wins.reg['start'].values).values

            inverted_wins.reg['new_stop'] = shifted_wins.reg.loc[inverted_win_prev, 'stop'].values
            missing = inverted_wins.classify_events(times, merge_wincols=('new_stop',))

            extra = missing['new_stop']
            shifted_times = pd.concat([shifted_times, extra])

            # preserve original order
            shifted_times.reindex(times.index.intersection(shifted_times.index))

        return shifted_times

    def defrag_series(self, traces: pd.DataFrame):
        """
        Take the given times and return them after shifting them so that all of the windows
        are consecutive.

        see defrag and defrag_events

        example:

            beta_compressed = sleep_wins.sel(cat='rem').defrag_series(beta)

        :param traces:
            Some signal with time as index.
            Either pd.Series or pd.DataFrame
        """
        sections = []

        cum_time = 0
        for idx in self.reg.index:
            section = traces.loc[self.reg.loc[idx, 'start']:self.reg.loc[idx, 'stop']]
            section = section.iloc[:-1].copy()
            section.index = section.index - self.reg.loc[idx, 'start'] + cum_time
            cum_time += self.reg.loc[idx, 'stop'] - self.reg.loc[idx, 'start']

            sections.append(section)

        compressed = pd.concat(sections)
        compressed.index.name = traces.index.name

        if isinstance(compressed, pd.Series):
            compressed.name = traces.name

        elif isinstance(compressed, pd.DataFrame):
            compressed.columns.name = traces.columns.name

        return compressed

    def _merge_consecutive(self, criteria, take='first'):
        """
        Merge multiple windows according to some pair-wise criteria.
        The current order of this windows (not of "criteria") is respected.
        Multiple consecutive windows can be merged in one, that will have the earliest
        start time and the latest stop time.
        For example:

            wins = timeslice.Windows.build_between(np.arange(10))
            wins.reg['cat'] = 'unknown'

            wins.reg.loc[2, 'cat'] = 'special'
            wins.reg.loc[6:7, 'cat'] = 'special'

            same_cat = wins.reg['cat'].iloc[:-1] == wins.reg['cat'].values[1:]

            merged_wins = _merge_consecutive(wins, same_cat)

            display(same_cat, wins, merged_wins)

        produces:
            wins:
                         start  stop  ref      cat
                win_idx
                0            0     1    0  unknown
                1            1     2    1  unknown
                2            2     3    2  special
                3            3     4    3  unknown
                4            4     5    4  unknown
                5            5     6    5  unknown
                6            6     7    6  special
                7            7     8    7  special
                8            8     9    8  unknown


            same_cat:
                    win_idx
                0     True
                1    False
                2    False
                3     True
                4     True
                5    False
                6     True
                7    False

            merged_wins:

                         start  stop  ref      cat
                win_idx
                0            0     2    0  unknown
                2            2     3    2  special
                3            3     6    3  unknown
                6            6     8    6  special
                8            8     9    8  unknown

        :param criteria: a boolean series indicating consecutive windows to merge.
        For example, if you do:

            criteria = sorted_wins['cat'].iloc[:-1] == sorted_wins['cat'].values[1:]

        This indicates whether any window i shares the same category as the next window i+1
        Note that because of being pairwise, we miss one element at the end.
        That element is still mergeable but only backwards (if second-to-last is True).

        :param take: whether the properties should be taken from the first or the last row of
        a set of multiple consecutive mergeable windows

        :return:
        """
        criteria = criteria.reindex(self.reg.index, fill_value=False)

        assert take in ['first', 'last']
        i = 0

        rows = {}

        while i < len(criteria):

            merge_start = None

            if not criteria.iloc[i]:
                rows[criteria.index[i]] = self.reg.iloc[i]
                i += 1

            else:

                while criteria.iloc[i] and i < len(criteria):
                    if merge_start is None:
                        merge_start = i
                    i += 1

                first_row = self.reg.iloc[merge_start]
                last_row = self.reg.iloc[i]

                idx = criteria.index[merge_start if take == 'first' else i]
                rows[idx] = self.reg.loc[idx].copy()
                rows[idx]['start'] = np.nanmin([first_row['start'], last_row['start']])
                rows[idx]['stop'] = np.nanmax([first_row['stop'], last_row['stop']])

                i += 1

        return self.__class__(pd.DataFrame.from_dict(rows, orient='index'))

    def merge_tight(self, same_cat=False, take='first'):
        """
        Merge any two windows that share an edge

        :param same_cat: whether to check the 'cat' column to determine if two windows can be merged
        :param take: take metadata from first or last window on merge
        :return:
        """
        sorted_wins = self.reg.sort_values(['start', 'stop'])

        mergeable = (sorted_wins['start'].iloc[:-1] - sorted_wins['stop'].values[1:]) <= 0

        if same_cat:
            same_cat_wins = sorted_wins['cat'].iloc[:-1] == sorted_wins['cat'].values[1:]
            mergeable = mergeable & same_cat_wins

        return self._merge_consecutive(mergeable, take=take)

    def merge_overlap(self, same_cat=False):
        """
        Merge any two windows that overlap.
        Any extra metadata will be taken from the earliest window.
        If same_cat=False, the resulting windows should be exclusive.

        :param same_cat: whether to check the 'cat' column to determine if two windows can be merged
        """
        # TODO change to use _merge_consecutive
        cat_col = 'cat'

        index = self.reg.sort_values('start').index

        pairs = []

        w_stop = index[0]
        w_start = w_stop

        for next_win in index[1:]:
            share_cat = True
            if same_cat:
                share_cat = self.reg.loc[next_win, cat_col] == self.reg.loc[w_stop, cat_col]

            next_range = self.reg.loc[next_win, ['start', 'stop']]
            current_stop = self.reg.loc[w_stop, 'stop']

            overlap = next_range['start'] <= current_stop < next_range['stop']

            if overlap and share_cat:
                w_stop = next_win

            else:
                pairs.append((w_start, w_stop))
                w_start = next_win
                w_stop = next_win

        pairs.append((w_start, w_stop))

        pairs = np.array(pairs).T

        merged = self.reg.loc[pairs[0]].copy()
        merged['stop'] = self.reg.loc[pairs[1], 'stop'].values

        merged = self.__class__(merged)

        if cat_col is None:
            assert merged.are_exclusive()

        if self.are_exclusive():
            assert np.isclose(self.total(), merged.total()), \
                f'Expected total time to match before ({self.total()}) and after merge ({self.total()})'

        return merged

    def overlap(self):
        """
        Return a boolean series indicating overlaps.
        Note that we are prioritising later windows.
        """
        self_sorted = self.rename_index('orig_idx').reset_index(drop=False).sort_values(['start', 'stop', 'orig_idx'])

        # TODO is this the same criterion as in pd.DataFrame.duplicates ?
        # noinspection PyTypeChecker
        overlapping = pd.Series(
            np.concatenate([self_sorted['stop'].values[:-1] > self_sorted['start'].values[1:], [False]]),
            index=self_sorted['orig_idx'].values
        )

        return overlapping.reindex(self.index)

    def drop_overlap(self, quiet=True):
        overlapping = self.overlap()

        if not quiet:
            print(
                'dropping',
                f'{np.count_nonzero(overlapping)}/{len(self.reg)} '
                f'({100 * np.count_nonzero(overlapping) / len(self.reg)}%)',
                'of windows because of overlap'
            )

        # TODO options?
        # We are prioritising later windows by this overlap detection procedure
        # we could instead split evenly the overlapping region (un-even windows, but keeping all)
        # or select random sets
        new = self.__class__(self.reg[~overlapping])

        assert new.are_exclusive()
        return new

    def drop_empty(self):
        return self.sel_mask(~self.is_empty())

    def drop_duplicates(self, subset=('start', 'stop')):
        return Windows(
            self.reg.drop_duplicates(subset=list(subset))
        )

    def prev_cat(self, dim='cat') -> pd.Series:
        """return the category of the previous window"""
        return self.reg[dim].shift(1)

    def next_cat(self, dim='cat') -> pd.Series:
        """return the category of the next window"""
        return self.reg[dim].shift(-1)

    def sandwiched(self, dim='cat', max_length=None, cat=None) -> pd.Series:
        """
        Return a boolean series indicating which windows are preceded and followed by the same category.

        :param dim:
        :param max_length: only apply to windows up to this length
        :param cat: only apply to windows of this category. A list allows for multiple categories.
        """
        next_cats = self.next_cat(dim)
        bad = (
                (next_cats == self.prev_cat(dim))
                &
                (next_cats != self.reg[dim])
        )

        if max_length is not None:
            bad = bad & (self.lengths() < max_length)

        if cat is not None:
            if not isinstance(cat, (list, tuple, np.ndarray)):
                cat = [cat]

            bad = bad & self.reg[dim].isin(cat)

        return bad

    def merge_sandwiched(self, dim='cat', max_length=None, cat=None):
        """
        Re-categorize and merge windows that live between two other windows of the same category.
        This is useful to remove small windows are created by noisy data after a simple thresholding.

        Note that it's possible that two consecutive windows may classify as "sandwiched",
        or that, even after merging, the resulting window is still sandwiched.
        This method will loop until all sandwiches have been merged,
        with a bias to taking the latter category.

        :param dim:
        :param max_length: only apply to windows up to this length
        :param cat: only apply to windows of this category

        :return: a new Windows object
        """
        wins = self.copy()

        are_sandwiched = wins.sandwiched(dim=dim, max_length=max_length, cat=cat)

        while are_sandwiched.any():
            # print(np.count_nonzero(are_sandwiched))
            wins.reg.loc[are_sandwiched, dim] = wins.next_cat(dim).loc[are_sandwiched].values
            are_sandwiched = wins.sandwiched(dim=dim, max_length=max_length, cat=cat)
            wins = wins.merge_tight(same_cat=True)

        return wins

    def invert(
            self,
            start=None,
            stop=None,
            keep_prev=tuple(),
            keep_next=tuple(),
            drop_empty=True,
    ):
        """
        Select all the time except that covered by the given windows.
        You can use this to define a baseline time period that is far enough from key events:

        # select all time in the series excluding any 500 ms window after an induced spike
        baseline_wins = spt.invert_windows(
            spt.make_windows(patched_spikes.time, (0, +500)),
            start=series_window[0],
            stop=series_window[1]
        )
        :param keep_prev:
        :param keep_next:
        :param drop_empty:
        :param start: if present, add a window between "start" to the start of the first window
        :param stop: if present, add a window between the stop of the last window to "stop"
        :return:
        """
        if not self.are_exclusive():
            windows = self.merge_overlap()
            assert self.are_exclusive()

        else:
            windows = self

        if len(windows) == 0:
            assert start is not None and stop is not None
            df = pd.DataFrame({'start': [start], 'stop': [stop], 'ref': [start]})
            df.index.name = 'win_idx'
            return self.__class__(df)

        windows = windows.reg.sort_values('start')

        df = {
            'start': windows['stop'].values[:-1],
            'stop': windows['start'].values[1:],
        }

        if start is not None:
            endpoint = windows.start.values[0]
            if start <= endpoint:
                df['start'] = np.append(start, df['start'])
                df['stop'] = np.append(endpoint, df['stop'])

        if stop is not None:
            endpoint = windows.stop.values[-1]
            if stop >= endpoint:
                df['start'] = np.append(df['start'], endpoint)
                df['stop'] = np.append(df['stop'], stop)

        if isinstance(keep_next, str):
            keep_next = [keep_next]

        for col in keep_next:
            df[f'next_{col}'] = windows[col].values[1:]

        if isinstance(keep_prev, str):
            keep_prev = [keep_prev]

        for col in keep_prev:
            df[f'prev_{col}'] = windows[col].values[:-1]

        df = pd.DataFrame(df)

        if drop_empty:
            df = df[df.start != df.stop].copy()

        df.sort_values('start', inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['ref'] = df['start']
        df.index.name = 'win_idx'

        return self.__class__(df)

    def classify_events(self, times, ref_col='ref', merge_wincols=None, right=False) -> pd.DataFrame:
        """
        compute the relative time of each event, depending on which window it falls into

        :param times:
        :param right: whether windows are right-inclusive (pre, post] or left [pre, post)

        :param merge_wincols:
            columns from windows that are copied over in the resulting DF
            defaults to 'cat' if present.

        :param ref_col: either the name of a window property or a list of them.
        if a list is provided, you get one delay with respect to each
        (for example delay_from_start and delay_from_stop).

        :returns: DF be like

                         delay  win_idx
            spike_idx
            1             58.7        0
            2             59.4        0
            ...            ...      ...
            248507       122.9       98
            248509       134.5       98

        Note that the index will be unique if the windows are exclusive.
        Otherwise the same spike may appear more than once.
        You can check this with:

            delays = wins.classify_events(spikes['time'])
            assert delays.index.is_unique

        """
        remaining = self

        # classification when windows are exclusive is fast
        # we're going to perform a classifications after dropping any window that overlaps
        # and then process the remaining windows, until there are none left.
        rounds = []
        while len(remaining) > 0:
            overlapping = remaining.overlap()

            ewins = Windows(remaining.reg[~overlapping])
            assert ewins.are_exclusive()

            rounds.append(_classify_events_exclusive(
                ewins.reg, times, ref_col,
                merge_wincols=merge_wincols, right=right))

            remaining = Windows(remaining.reg[overlapping])

        return pd.concat(rounds, axis=0)

    def annotate_events(self, events: pd.DataFrame, prefix=None, col='time', drop=False,
                        ref_col='ref', merge_wincols=None, right=False) -> pd.DataFrame:
        """
        Classify the events and according to these windows and add information about the classification
        to produce a unified table of metadata.

        :param events:
        :param prefix:
        :param col: by default classify on "time"
        :param drop: whether spikes outside the windows should be dropped.

        :param ref_col:
        :param merge_wincols:
        :param right:

        :return: a copy of "events" with extra information from the classification
         (delay, win_idx, cat and any other window metadata)
        """
        assert self.are_exclusive()

        if merge_wincols is None:
            merge_wincols = list(self.reg.columns.difference(['start', 'stop', 'ref']))

        classified = self.classify_events(
            events[col],
            ref_col=ref_col,
            merge_wincols=merge_wincols,
            right=right,
        )

        if prefix is not None:
            classified = classified.add_prefix(f'{prefix}_')

        overlapping_cols = classified.columns.intersection(events.columns)

        if len(overlapping_cols) > 0:
            logging.warning(f'{len(overlapping_cols)} columns already exist: {list(overlapping_cols)}')

        merged = pd.merge(
            events,
            classified,
            left_index=True,
            right_index=True,
            how='inner' if drop else 'left',
        )

        return merged

    def classify_windows(self, others, align_to='ref', edges='crop') -> dict:
        """
        Classify another set of windows within these,
        computing the relative time of start/stop/ref, and cropping if necessary.

        For example, if you have a set of large windows around some stimuli
        and a different set of smaller windows that define behavioral states, you can
        classify the states by the stimuli. Like:

            pulse_zooms = Windows.build_centered(pulses, ms(minutes=10))
            classified_rem_wins = pulse_zooms.classify_windows(rem_wins, name='pulse_idx')

        :returns: a dict from this window indices to a Windows object of other after cropping
            Note that, if "self" is overlapping, then there may be more windows returned than given.
        """

        assert edges in ('crop', 'drop', 'keep')

        if align_to is not None:
            ref_times = self.relative_time(align_to)
        else:
            ref_times = None

        results = {}

        for idx, zoom in self.iter_wins():

            if edges == 'crop':
                zoomed = others.crop_to_main(zoom)

            else:
                if edges == 'drop':
                    fully = others.contained_in(zoom, fully=True)
                    zoomed = others.sel_mask(fully)

                else:
                    assert edges == 'keep'

                    partial = others.contained_in(zoom, fully=False)
                    zoomed = others.sel_mask(partial)

            zoomed = zoomed.drop_empty()

            if ref_times is not None:
                zoomed = zoomed.shift(-ref_times[idx])

            results[idx] = zoomed

        return results

    def get_rolled(self, col, roll_win, on='ref', sampling=101) -> pd.Series:
        x = self[on]
        y = self[col]

        if isinstance(sampling, int):
            new_x = np.linspace(self[on].min(), self[on].max(), sampling)

        else:
            new_x = sampling

        return mean_roll_by(
            x, y,
            Win.build_centered(0, roll_win),
            new_x,
        )

    def get_rolled_multiple(self, cols, roll_win, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({
            col: self.get_rolled(col=col, roll_win=roll_win, **kwargs)
            for col in cols
        })

    def get_inter_intervals(
            self,
            first='stop', second='start', sortby='ref',
            ascending=True,
            shift=1,
    ) -> pd.Series:
        """
        Get the duration from "this" window to the "next" window.
        :param first: what time stamp to consider for "this" window (default: stop)
        :param second: what time stamp to consider for the "next" window (default: start)
        :param sortby: to determine who is next, pre-sort windows by this time stamp.
        :param ascending: sort ascending (see sortby)
        :param shift: how many windows to skip. Usually you'll just want to compare one to the next (+1, default).
        :return:
        """
        assert shift >= 0
        wins = self.sort_values(sortby, ascending=ascending)
        td = wins.relative_time(second).shift(-shift) - wins.relative_time(first)
        td = td.reindex(self.index)
        return td

    def get_inter_intervals_by(
            self,
            cat0, cat1,
            by='cat',
            first='ref', second='ref', sortby='ref', ascending=True,
            reindex=True,
    ) -> pd.Series:
        """
        Get the duration from "this" window to the "next" window,
        but only if "this" is of category cat0 and "next" is of category cat1.

        This is useful to compute inter-event intervals across two channels::
            events.get_inter_intervals(0, 1, by='channel)

        see get_inter_intervals

        :param cat0: inter intervals are extracted only for this type of window
        :param cat1: inter intervals are extracted relative to this type of window
        :param by: what column to use as categories
        :param reindex: if true, windows of categories different of cat0 will contain NaN.
            Otherwise, series may be shorter than the total number of windows.
        :param first: what time stamp to consider for "this" window (default: stop)
        :param second: what time stamp to consider for the "next" window (default: start)
        :param sortby: to determine who is next, pre-sort windows by this time stamp.
        :param ascending: sort ascending (see sortby)
        :return:
        """

        reg = self.reg.sort_values(sortby, ascending=ascending)
        reg = reg.loc[reg[by].isin([cat0, cat1])]

        consecutive = (
                (reg[by].values[:-1] == cat0) &
                (reg[by].values[1:] == cat1)
        )
        first_idcs = reg.index[:-1][consecutive]
        second_idcs = reg.index[1:][consecutive]

        assert len(first_idcs) == len(second_idcs)

        sec = self.relative_time(second)
        fir = self.relative_time(first)

        td = sec.loc[second_idcs].values - fir.loc[first_idcs].values
        td = pd.Series(td, first_idcs)

        if reindex:
            td = td.reindex(self.reg.index)

        return td

    def get_histogram(self, name, bins=100) -> pd.Series:

        if isinstance(bins, int):
            bins = np.linspace(self[name].min(), self[name].max(), bins + 1)

        h, bins = np.histogram(self[name], bins=bins)

        return pd.Series(h, index=pd.IntervalIndex.from_breaks(bins))

    def get_histogram_length(self, bins) -> pd.Series:
        this = self.copy()
        this['length'] = self.lengths()
        return this.get_histogram('length', bins)

    def get_histograms_by(self, name, bins=100, by='cat') -> pd.DataFrame:

        if isinstance(bins, int):
            bins = np.linspace(self[name].min(), self[name].max(), bins + 1)

        dists = {
            cat: np.histogram(wins[name], bins=bins)[0]
            for cat, wins in self.iter_groupby(by)
        }

        dists = pd.DataFrame(dists, index=pd.IntervalIndex.from_breaks(bins))

        return dists

    def get_histograms_length_by(self, bins, by='cat') -> pd.DataFrame:
        this = self.copy()
        this['length'] = self.lengths()
        return this.get_histograms_by('length', bins, by=by)

    def is_isolated(self, at_least: float) -> pd.Series:

        if isinstance(at_least, (tuple, list)):
            pre, post = at_least
        else:
            pre = at_least
            post = at_least

        to_prev = self.interval_to_prev()
        to_next = self.interval_to_next()

        # noinspection PyTypeChecker
        return (to_next >= post) & (to_prev >= pre)

    def interval_to_closest(self) -> pd.Series:
        to_prev: pd.Series = self.interval_to_prev()
        to_next: pd.Series = self.interval_to_next()
        return np.minimum(to_next, to_prev)

    def interval_to_prev(self, shift=1) -> pd.Series:
        to_prev: pd.Series = self.get_inter_intervals(first='start', second='stop', ascending=False, shift=shift) * -1
        to_prev.fillna(np.inf, inplace=True)
        return to_prev

    def interval_to_next(self, shift=1) -> pd.Series:
        to_next: pd.Series = self.get_inter_intervals(first='stop', second='start', ascending=True, shift=shift)
        to_next.fillna(np.inf, inplace=True)
        return to_next

    def extend(self, pre=0., post=0.):
        """
        Add (or remove) time at the start (pre) or the end (post) of the window
        :param pre: time in milliseconds relative to start
        :param post: time in milliseconds relative to stop
        :return: a new tuple object
        """
        copy = self.copy()
        copy.reg['start'] += to_ms(pre)
        copy.reg['stop'] += to_ms(post)

        return copy


def _classify_events_exclusive(
        windows: pd.DataFrame,
        times,
        ref_col,
        merge_wincols,
        right,
):
    """
    compute the relative time of each event, depending on which window it falls into
    see Windows.classify_events
    """

    ######################################################################
    # clean arguments

    if merge_wincols is None:
        if 'cat' in windows.columns:
            merge_wincols = ['cat']
        else:
            merge_wincols = []

    merge_wincols = list(merge_wincols)

    # non-empty, monotonically increasing, windows
    windows = windows.sort_values(['start', 'stop'])
    lengths = windows['stop'] - windows['start']
    windows = windows[lengths > 0]

    # array of [start, stop, start, stop, ..., start, stop]
    edges = windows[['start', 'stop']].values.flatten()

    if not isinstance(times, pd.Series):
        times = pd.Series(np.asarray(times))

    if isinstance(ref_col, str):
        ref_col = [ref_col]

    ts = times.values

    ######################################################################
    # pure numpy classification

    # will return the index so that
    # edges[i-1] <= t < edges[i]
    edge_idx_per_t = np.digitize(ts, edges, right=right)

    # shift idcs to represent left edge so:
    # edges[i] <= t < edges[i+1]
    edge_idx_per_t = edge_idx_per_t - 1

    valid = (
        # didn't fall outside our covered time
            (0 <= edge_idx_per_t) & (edge_idx_per_t < len(edges) - 1) &

            # only even pairs pair of values in "edges" are actual windows:
            # [start, stop, start, stop, ..., start, stop]
            ((edge_idx_per_t % 2) == 0)
    )

    # we don't care about rounding for odd numbers,
    # because those represent "stop" and should be marked as invalid
    win_idx_per_t = edge_idx_per_t // 2

    ######################################################################
    # construct DF with results

    times = times[valid]
    win_idx_per_t = win_idx_per_t[valid]

    win_ids = windows.index[win_idx_per_t]

    delays = times.values[np.newaxis, :] - windows[ref_col].iloc[win_idx_per_t].values.T

    wincol_name = windows.index.name
    if wincol_name is None:
        wincol_name = 'win_idx'

    df = {
        wincol_name: win_ids,
    }

    for name, d in zip(ref_col, delays):
        df[f'delay_from_{name}' if name != 'ref' else 'delay'] = d

    df = pd.DataFrame(df, index=times.index)

    if df.index.name == windows.index.name:
        # drop the index name if it can create a conflict
        # because pandas doesn't like merging two dfs where indicies and columns
        # are called the same, even though we specify we want to merge on the index only
        df.index.name = None

    df = pd.merge(
        df,
        windows[merge_wincols],
        left_on=wincol_name,
        right_index=True)

    return df


def get_stride(stored_hz, load_hz):
    """
    Compute the best stride to load data (length of each jump) when the data was stored at a certain
    sampling rate but we want to load at a different one.

    Note this will not respect exactly the load_hz, but return the stride corresponding to the closest one.
    """
    return int(np.round(stored_hz / load_hz))


def match_load_hz(stored_hz, load_hz, thresh=None):
    """Adjust the load_hz to produce a perfect integer stride"""

    new_load_hz = (stored_hz / get_stride(stored_hz, load_hz))

    valid = (
        abs(new_load_hz - load_hz) < thresh
        if thresh is not None else
        np.isclose(new_load_hz - load_hz, 0)
    )

    if not valid:
        logging.warning(f'Adjusting load_hz from {load_hz}Hz to {new_load_hz}Hz to make it '
                        f'a perfect divisor of stored_hz {stored_hz}Hz (stride: {get_stride(stored_hz, new_load_hz)})')

    return new_load_hz


def check_stride(sampling_hz: float, downsample_hz: float):
    """check that the downsample_hz is as close as possible to a perfect divisor of the sampling  rate"""
    return np.isclose(get_stride(sampling_hz, downsample_hz), (sampling_hz / downsample_hz))


def assert_stride(sampling_hz, downsample_hz, numerator_name='sampling_hz', denominator_name='downsample_hz'):
    """assert if the downsample_hz is not a divisor of the sampling  rate"""
    assert check_stride(sampling_hz, downsample_hz), \
        f'Expected {denominator_name} ({downsample_hz}) to be divisor of {numerator_name} ({sampling_hz})'


def warn_stride(sampling_hz, downsample_hz, numerator_name='sampling_hz', denominator_name='downsample_hz'):
    """warn if the downsample_hz is not a divisor of the sampling  rate"""
    if not check_stride(sampling_hz, downsample_hz):
        logging.warning(
            f'Expected {denominator_name} ({downsample_hz}) to be divisor of {numerator_name} ({sampling_hz})')


def strfdelta(tdelta, plus_sign=False, strip=True, show_days=False):
    """

    pretty str format a timedelta object
    accepts negative values!

    :param tdelta:
    :param show_days:
    :param plus_sign: whether a '+' should be place if the timedelta is positive.
    This can be useful when we want to ensure a fixed length of the string.

    :param strip:
        If true don't plot seconds or milliseconds if they are 0.
    :return:
    """
    total = tdelta.total_seconds()

    sign = '' if not plus_sign else '+'
    if total < 0:
        sign = '-'
        total = total * -1

    if show_days:
        days, hours = divmod(total, 60 * 60 * 24)
    else:
        days = None
        hours = total

    hours, minutes = divmod(hours, 60 * 60)
    minutes, seconds = divmod(minutes, 60)
    seconds, decimals = divmod(seconds, 1)

    desc = ''
    if show_days:
        desc = f'{days:g}d '

    desc = desc + f'{sign}{int(hours):02d}:{int(minutes):02d}'

    if seconds > 0 or decimals > 0 or not strip:
        desc += f':{seconds:02.0f}'

        if decimals or not strip:
            desc += f'.{decimals * 1000.:03.0f}'

    return desc


def strf_ms(value, plus_sign=False, strip=True, show_days=False):
    """
    pretty str format a float value representing milliseconds
    """
    return strfdelta(
        timedelta(milliseconds=to_ms(value)),
        plus_sign=plus_sign,
        strip=strip,
        show_days=show_days,
    )


def milliseconds_to_timestamp(milliseconds: float):
    """
    convert total milliseconds to a formatted timestamp string DDd HH:MM:SS.sss

    This is meant for human-readable serialization.
    For pretty-printing with more options see strf_ms
    """

    seconds, mils = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    return f"{int(days):02}d {int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(mils):03}"


def timestamp_to_milliseconds(timestamp) -> float:
    """
    convert a timestamp string DDd HH:MM:SS.sss to total milliseconds
    Days, seconds and milliseconds are optional
    examples: ["3d 19:00:15.143", "19:00", "3d 19:00", "19:00:15"]
    """
    import re
    # Define regex pattern with optional days, hours, minutes, optional seconds, and milliseconds
    pattern = r'(?:(\d+)d)?\s*(?:(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?)?'
    match = re.match(pattern, timestamp)

    if not match:
        raise ValueError("Invalid timestamp format")

    # Extract matched groups
    days = int(match.group(1)) if match.group(1) else 0
    hours = int(match.group(2)) if match.group(2) else 0
    minutes = int(match.group(3)) if match.group(3) else 0
    seconds = int(match.group(4)) if match.group(4) else 0
    milliseconds = int(match.group(5).ljust(3, '0')) if match.group(5) else 0

    return ms(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )


def get_solar_offset(first_timestamp, hours=0, minutes=0) -> float:
    """
    Given a timestamp of when a recording started, calculate the offset in milliseconds
    from midnight (or from the given hours:minutes).
    This can be used to transport data from a recording timeframe to a solar timeframe.
    """
    zeitgeber = first_timestamp.replace(hour=hours, minute=minutes, second=0, microsecond=0)

    time_offset = (first_timestamp - zeitgeber).total_seconds() * 1000

    return time_offset


def time_to_circadian(t_ms, first_timestamp):
    """adjust a ms timepoint to a ms timepont where t=0 is 7am (this is a circadian convention)"""
    time_offset = get_solar_offset(first_timestamp, hours=7, minutes=0)
    return t_ms + time_offset


def time_to_solar(t_ms, first_timestamp):
    """convert a ms timepoint to solar datetime object"""
    time_offset = get_solar_offset(first_timestamp, hours=0, minutes=0)
    return t_ms + time_offset


def solar_to_time(t_ms, first_timestamp):
    when = first_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(milliseconds=t_ms)

    return to_ms(when - first_timestamp)


def build_light_dark_cycle(within, lights_on=ms(hours=7), lights_off=ms(hours=19)) -> Windows:
    """
    Build light/dark windows within the given time period, assuming lights are always
    turned off/on at the same solar times
    """
    cycles = []

    one_day = ms(days=1)

    start_in_the_day = within.start % one_day
    current_state = "on" if lights_on <= start_in_the_day < lights_off else "off"

    current_time = within.start

    while current_time < within.stop:
        if current_state == "on":
            next_time = (current_time // one_day) * one_day + lights_off
        else:
            next_time = (current_time // one_day + 1) * one_day + lights_on

        if next_time > within.stop:
            next_time = within.stop

        cycles.append((current_state, current_time, next_time))

        current_time = next_time
        current_state = "off" if current_state == "on" else "on"

    cycles = pd.DataFrame(cycles, columns=['cat', 'start', 'stop'])

    return Windows(cycles)


def strf_circ(t_ms, first_timestamp, drop_ms=True) -> str:
    """pretty print a ms timepoint using circadian convetion"""
    solar = time_to_solar(t_ms, first_timestamp)
    if drop_ms:
        solar = solar.replace(microsecond=0)

    circ = time_to_circadian(t_ms, first_timestamp)
    if drop_ms:
        circ = np.round(circ * MS_TO_S) * S_TO_MS

    return f'{solar} (ZT{strf_ms(circ, plus_sign=True)})'


def ms_to_idcs(sampling_rate: float, time_ms: np.array):
    return np.round(sampling_rate * time_ms * MS_TO_S).astype(int)


def idcs_to_ms(sampling_rate: float, idcs: np.array):
    return (idcs / sampling_rate) / MS_TO_S


def adjust_sampling_period(period, quiet=False):
    # time unit is milliseconds, so we are going
    # to round up to 1 pico second
    new = np.round(period, decimals=9)

    if not quiet and not np.isclose(new, period):
        logging.warning(f'Adjusting sampling period from {period} to {new}')

    return new


def adjust_to_sampling_period(length, period, desc=None):
    if isinstance(length, tuple):
        assert len(length) == 2
        return Win(
            adjust_to_sampling_period(length[0], period, desc=f'{desc} start' if desc is not None else None),
            adjust_to_sampling_period(length[1], period, desc=f'{desc} stop' if desc is not None else None),
        )

    else:

        new = np.round(length / period) * period

        if desc is not None and not np.isclose(new, length):
            logging.warning(f'Adjusting {desc} from {length} to {new}')

        return new
