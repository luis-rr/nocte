import functools
import logging

import numba as nb
import numpy as np
import pandas as pd
import scipy.signal
from tqdm.auto import tqdm

from nocte import datadict as dd
from nocte import timeslice
from nocte.analysis import sleep
from nocte.df_wrapper import DataFrameWrapper, _optional_pbar
from typing import Self


# @nb.njit(parallel=True)
def _cross_corr_shifted_nb(
        signal0: np.ndarray,
        signal1: np.ndarray,
        idcs: np.ndarray,
        offset: int,
):
    """
    Fast sliding window cross corr.
    See parallelization diagnostics like:
        xc._cross_corr_nb.parallel_diagnostics(level=4)

    :param signal0:
    :param signal1:

    :param idcs:
        shape <N, 2> indicating the pair of indices to cut c0 and c1 to comput xcorr.
        c1 indices will additionally be offset by "offset".

    :param offset:

    :return: array of same length as idcs0
    """
    corr = np.ones(len(idcs)) * np.nan

    idcs_shifted = idcs + offset

    valid = (idcs_shifted >= 0) & (idcs_shifted <= len(signal1))
    valid = valid[:, 0] & valid[:, 1]

    # assert np.min(idcs_shifted[:, 0]) >= 0, 'must zero-pad array or crop windows'
    # assert np.max(idcs_shifted[:, 1]) < len(signal1), 'must zero-pad array or crop windows'

    valid_idcs, = np.where(valid)
    iter_count = len(valid_idcs)

    for j in nb.prange(iter_count):
        i = valid_idcs[j]

        start0, stop0 = idcs[i]
        start1, stop1 = idcs_shifted[i]

        section0 = signal0[start0:stop0]
        section1 = signal1[start1:stop1]

        corr[i] = np.sum(section0 * section1)

    return corr


@nb.njit(parallel=True)
def _cross_corr_shifted_pearsons_nb(
        signal0: np.ndarray,
        signal1: np.ndarray,
        idcs: np.ndarray,
        offset: int,
):
    """
    Fast sliding window cross corr.
    See parallelization diagnostics like:
        xc._cross_corr_nb.parallel_diagnostics(level=4)

    :param signal0:
    :param signal1:

    :param idcs:
        shape <N, 2> indicating the pair of indices to cut c0 and c1 to comput xcorr.
        c1 indices will additionally be offset by "offset".

    :param offset:

    :return: array of same length as idcs0
    """
    corr = np.ones(len(idcs)) * np.nan

    idcs_shifted = idcs + offset

    valid = (idcs_shifted >= 0) & (idcs_shifted <= len(signal1))
    valid = valid[:, 0] & valid[:, 1]

    # assert np.min(idcs_shifted[:, 0]) >= 0, 'must zero-pad array or crop windows'
    # assert np.max(idcs_shifted[:, 1]) < len(signal1), 'must zero-pad array or crop windows'

    valid_idcs, = np.where(valid)
    iter_count = len(valid_idcs)

    for j in nb.prange(iter_count):
        i = valid_idcs[j]

        start0, stop0 = idcs[i]
        start1, stop1 = idcs_shifted[i]

        section0 = signal0[start0:stop0]
        section1 = signal1[start1:stop1]

        section0 = (section0 - np.mean(section0)) / np.std(section0)
        section1 = (section1 - np.mean(section1)) / np.std(section1)

        corr[i] = np.mean(section0 * section1)

    return corr


def _assert_multiple(sampling_period_ms, t):
    is_multiple = (t % sampling_period_ms == 0)

    if isinstance(is_multiple, np.ndarray):
        is_multiple = np.all(is_multiple)

    assert is_multiple, \
        f'Signal sampled every {sampling_period_ms} ms, ' \
        f'but asking for non-multiple {t} ms'


# @nb.njit(parallel=True)
def _rolling_cross_corr_discreete(
        s0: np.ndarray,
        s1: np.ndarray,
        offsets: np.ndarray,
        sliding_win: int,
        sliding_step: int,
        pbar=None,
        pearson=False,
):
    length = min(len(s0), len(s1))

    starts = np.arange(0, length - sliding_win + 1, sliding_step)
    stops = np.arange(sliding_win, length + 1, sliding_step)
    sliding_wins = np.array([starts, stops]).T

    xcorr = np.empty((len(offsets), len(sliding_wins)))

    for i, offset in enumerate(_optional_pbar(offsets, pbar=pbar, total=len(offsets), desc='lag')):

        if pearson:
            value = _cross_corr_shifted_pearsons_nb(s0, s1, sliding_wins, offset=offset)

        else:
            value = _cross_corr_shifted_nb(s0, s1, sliding_wins, offset=offset)

        xcorr[i] = value

    return xcorr


def _rolling_cross_corr_ms(
        s0: np.ndarray,
        s1: np.ndarray,
        sampling_period_ms: float,
        lags_ms: np.ndarray,
        sliding_win_ms: float,
        sliding_step_ms: float,
        pbar=None,
        pearson=False,
):
    lags_ms = np.asarray(lags_ms)
    assert len(lags_ms) > 0

    _assert_multiple(sampling_period_ms, lags_ms)
    offsets = np.round(lags_ms / sampling_period_ms).astype(int)

    _assert_multiple(sampling_period_ms, sliding_win_ms)
    sliding_win = int(sliding_win_ms // sampling_period_ms)

    if sliding_step_ms is None:
        sliding_step_ms = sampling_period_ms
    _assert_multiple(sampling_period_ms, sliding_step_ms)
    sliding_step = int(sliding_step_ms // sampling_period_ms)

    xcorr = _rolling_cross_corr_discreete(
        s0,
        s1,
        offsets,
        sliding_win,
        sliding_step,
        pbar,
        pearson=pearson,
    )

    return xcorr


def _estimate_sampling_period(times, atol=1.e-6) -> float:
    dts = np.diff(times)
    dts = np.unique(dts)

    assert np.allclose(dts[0], dts, atol=atol), dts

    # noinspection PyTypeChecker
    return dts[0]


class Traces(DataFrameWrapper):
    """
    Class for storing time series as a pd.DataFrame (self.traces) with
    associated metadata as another pd.DataFrame (self.reg).
    """

    def __init__(
            self,
            reg: pd.DataFrame,
            traces: pd.DataFrame,
            copy=True,
    ):
        if copy:
            traces = traces.copy()
            reg = reg.copy()

        super().__init__(reg.copy())
        self.traces: pd.DataFrame = traces

        assert isinstance(reg, pd.DataFrame)
        assert reg.index.is_unique
        assert reg.columns.is_unique

        if self.reg.index.name is None:
            default_index_name = 'trace_idx'
            if default_index_name in reg.columns:
                logging.warning(f'Default index name "{default_index_name}" already in columns. Drop or rename first?')
            self.reg.rename_axis(index=default_index_name, inplace=True)

        assert self.traces.index.is_unique
        assert self.traces.columns.is_unique
        if self.traces.index.name is None:
            self.traces.rename_axis(index='time', inplace=True)
        if self.traces.columns.name is None:
            self.traces.rename_axis(columns=self.reg.index.name, inplace=True)

        assert self.traces.columns.name == self.reg.index.name

        assert len(self.reg.index) == len(self.traces.columns), \
            f'Got {len(self.reg.index)} reg entries but {len(self.traces.columns)} traces'

        assert np.all(self.reg.index == self.traces.columns)

        self.traces.columns.name = self.reg.index.name

    @classmethod
    def load_single(cls, loader, load_win, ref=None, load_hz=None, channels=None, pbar=None):

        load_win = timeslice.Win(*load_win)

        if ref is None:
            load_wins = timeslice.Windows.build_around(
                [load_win.start],
                (0, load_win.length),
            )
        else:
            load_wins = timeslice.Windows.build_around(
                pd.Series([ref]),
                load_win
            )

        result = cls.load_many(
            loader,
            load_wins,
            load_hz=load_hz,
            channels=channels,
            pbar=pbar,
        )

        if 'win_idx' in result.reg.columns:
            result.reg.drop('win_idx', axis=1, inplace=True)

        return result

    @classmethod
    def load_many(cls, loader, load_wins, load_hz=None, channels=None, pbar=None):

        if load_hz is None:
            load_hz = loader.sampling_rate

        if channels is None:
            channels = loader.channels.index

        traces = {}

        for idx, ref, win_ms in load_wins.iter_wins_ref(pbar=pbar):
            win_ms = win_ms.clip(loader.win_ms)

            win_ms_rel = win_ms.shift(-ref)
            slice_idcs_rel: slice = win_ms_rel.to_slice_idx(loader.sampling_rate, load_hz)

            # Rounding for closest sample together with imperfect sampling rates may cause
            # an off by one difference between:
            #   win_rel.to_slice_idx(loader.sampling_rate, load_hz)
            #   win.to_slice_idx(loader.sampling_rate, load_hz)
            # To prevent this, let's calculate the final idcs as a shift of the relative idcs
            # which we take as final truth.

            ref_idx = timeslice.SamplingRate(loader.sampling_rate).ms_to_idcs(ref)
            slice_idcs = slice(
                slice_idcs_rel.start + ref_idx,
                slice_idcs_rel.stop + ref_idx,
                slice_idcs_rel.step,
            )

            assert (slice_idcs.stop - slice_idcs.start) == (slice_idcs_rel.stop - slice_idcs_rel.start)

            # This is equivalent of converting to idcs in the ideal load_hz
            rel_idcs = np.arange(
                slice_idcs_rel.start / slice_idcs_rel.step,
                slice_idcs_rel.stop / slice_idcs_rel.step,
                1,
            )

            data = loader.load(slice_idcs, channels)

            data = pd.DataFrame(data.T, index=rel_idcs, columns=channels)
            data.rename_axis('channel', axis=1, inplace=True)

            traces[idx] = data

        wins_idx_name = load_wins.index.name

        traces = pd.concat(traces, axis=1, names=[wins_idx_name])

        traces.sort_index(inplace=True)
        assert traces.index.is_unique

        traces.index = traces.index.astype(int)

        traces.index = (traces.index * 1_000. / load_hz)

        new_reg = traces.columns.to_frame(index=False)
        merged_reg = pd.merge(
            new_reg,
            load_wins.wins.drop(['start', 'stop'], axis=1),
            how='left',
            left_on=wins_idx_name,
            right_index=True,
        )
        assert merged_reg.index.is_unique

        traces.columns = merged_reg.index

        return cls.from_df(
            reg=merged_reg,
            traces=traces,
        )

    @classmethod
    def from_multiindex_df(cls, df: pd.DataFrame):
        desc = df.columns.to_frame(index=False)
        traces = df.copy()
        traces.columns = desc.index

        return cls.from_df(
            traces=traces,
            reg=desc,
        )

    @classmethod
    def from_df(cls, traces: pd.DataFrame, reg: pd.DataFrame = None):
        """
        From a dataframe where index indicates time in milliseconds.
        Optionally provide extra info for the registry (desc), whose index must match the df columns.
        """

        traces = traces.copy()

        if reg is None:
            reg = traces.columns.to_frame(index=False)
            traces.columns = reg.index

        traces.sort_index(inplace=True)

        reg = reg.copy()

        return cls(
            reg=reg,
            traces=traces,
        )

    @classmethod
    def from_series(cls, s: pd.Series, col_name=None, entry_name=None):

        if entry_name is not None:
            s = s.rename(entry_name)

        df = s.to_frame()

        if col_name is not None:
            df.rename_axis(columns=col_name, inplace=True)

        return cls.from_df(df)

    @classmethod
    def from_dict(cls, d: dict, names: list):
        df = pd.concat(d, axis=1, names=names)
        return cls.from_df(df)

    @classmethod
    def from_dict_resampled(
            cls,
            d: dict,
            names: list,
            start='milliseconds',
            stop=None,
            period=None,
            reg: pd.DataFrame = None,
            pbar=None,
    ):
        assert len(d) > 0

        if start is None:
            start = min(trace.index.min() for k, trace in d.items())

        elif isinstance(start, str):
            vmin = min(trace.index.min() for k, trace in d.items())
            round_to = timeslice.ms(**{start: 1})
            start = np.floor(vmin / round_to) * round_to

        if stop is None:
            stop = max(trace.index.max() for k, trace in d.items())

        if period is None:
            period = min(np.min(np.diff(trace.index)) for k, trace in d.items())
            period = np.ceil(period * 0.5)

        logging.info(f'resampling from {start} to {stop} at {period}')

        win = timeslice.Win(start, stop)

        resampled = {}
        for k, trace in _optional_pbar(d.items(), total=len(d), pbar=pbar):

            if isinstance(trace, pd.Series):
                resampled[k] = win.interp_series(trace, step=period)
            else:
                assert isinstance(trace, pd.DataFrame)
                resampled[k] = win.interp_df(trace, step=period)

        lengths = np.array([trace.shape[0] for trace in resampled.values()])
        assert np.all(lengths[0] == lengths)

        time = pd.Index(list(resampled.values())[0].index)
        for df in resampled.values():
            df.reset_index(drop=True, inplace=True)

        resampled = pd.concat(
            resampled,
            axis=1,
            verify_integrity=False,
            sort=False,
            names=names,
        )
        resampled.index = time

        return cls.from_df(resampled, reg=reg)

    @classmethod
    def from_data_dict(cls, datadict: dd.DataDict, key_name: str = None, pre_aligned=False):
        """Assuming each entry is a traces object"""

        if key_name is None:
            assert datadict.index.name is not None
            key_name = datadict.index.name

        if pre_aligned:
            traces = cls.concat_dict_aligned(datadict.data, names=[key_name])

        else:
            traces = cls.concat_dict(datadict.data, key_name=[key_name])

        traces = traces.merge_reg(datadict.reg, left_on=[key_name], right_index=True)

        return traces

    def store_hdf(self, path, key='traces'):
        path = str(path)
        self.reg.to_hdf(path, key=f'{key}_reg')
        self.traces.to_hdf(path, key=f'{key}_data')

    @classmethod
    def load_hdf(cls, path, key='traces'):
        path = str(path)
        # noinspection PyTypeChecker
        return cls(
            reg=pd.read_hdf(path, key=f'{key}_reg'),
            traces=pd.read_hdf(path, key=f'{key}_data'),
            copy=False,
        )

    def to_dict(self, col):
        """
        Split this traces object into multiple ones with the key being the given column values.
        """
        return dict(self.iter_grouped(col))

    def first_valid_index(self):
        return self.traces.apply(lambda col: col.first_valid_index())

    def last_valid_index(self):
        return self.traces.apply(lambda col: col.last_valid_index())

    def to_wins(self, ref='ref', tight=True) -> timeslice.Windows:

        reg = self.reg.copy()
        if tight:
            start = self.first_valid_index()
            stop = self.last_valid_index()
        else:
            rel_win = self.get_global_win()
            start, stop = rel_win.start, rel_win.stop


        if ref in self.columns:
            refs = reg[ref]
        else:
            refs = 0

        reg['start'] = refs + start
        reg['stop'] = refs + stop

        return timeslice.Windows(reg)

    @classmethod
    def concat_dict(cls, traces_dict: dict, key_name=None):

        reg = pd.concat({
            k: traces.reg
            for k, traces in traces_dict.items()
        }, axis=0, names=key_name)

        reg.reset_index(inplace=True)
        reg.rename(columns=dict(trace_idx='local_trace_idx'), inplace=True)

        traces = pd.concat([
            traces.traces
            for traces in traces_dict.values()
        ], axis=1)

        traces.columns = reg.index

        return cls(
            reg=reg,
            traces=traces,
            copy=False,
        )

    @classmethod
    def concat_list(cls, traces_list: list):

        reg = pd.concat([traces.reg for traces in traces_list], axis=0)
        reg = reg.reset_index(drop=True)

        traces = pd.concat([traces.traces for traces in traces_list], axis=1)
        traces = traces.T.reset_index(drop=True).T
        traces.sort_index(inplace=True)

        return cls(
            reg=reg,
            traces=traces,
            copy=False,
        )

    @classmethod
    def concat_dict_aligned(cls, traces_dict: dict, names):
        """
        Concatenates multiple Traces objects assuming indices are regular and overlapping.
        Sampling rates must match and sampling time must be aligned, but data may be missing
        for some objects either at the beginning or at the end (longer or shorter recordings).

        This function pads shorter traces objects with NaN rows so they all have the same index as the longest one.

        This is much faster than concatenating DataFrames with pd.concat, which involves slow reindexing.
        """

        sampling_periods = np.array([traces.sampling_period for traces in traces_dict.values()])
        step = sampling_periods[0]
        assert np.all(sampling_periods == step), \
            f'Traces with different sampling periods'

        starts = np.array([traces.time[0] for traces in traces_dict.values()])
        global_start = np.min(starts)
        assert np.all(((starts - global_start) % sampling_periods[0]) == 0), \
            f'Traces sampling is misaligned'

        stops = np.array([traces.time[-1] for traces in traces_dict.values()])
        global_stop = np.max(stops)

        unified_index = pd.Index(np.arange(
            global_start,
            global_stop + step,
            step,
        ))

        padded_dfs = []
        for traces in traces_dict.values():
            df = traces.traces
            start = df.index[0]
            stop = df.index[-1]
            cols = df.columns

            start_pad = pd.DataFrame(np.nan, index=np.arange(global_start, start, step), columns=cols)
            stop_pad = pd.DataFrame(np.nan, index=np.arange(stop + step, global_stop + step, step), columns=cols)

            padded = pd.concat([start_pad, df, stop_pad])
            assert len(padded) == len(unified_index)
            padded_dfs.append(padded)

        combined_reg = pd.concat({
            k: traces.reg
            for k, traces in traces_dict.items()
        }, axis=0, names=names)

        combined_reg.reset_index(inplace=True, drop=False)
        combined_reg.drop('trace_idx', axis=1, inplace=True)

        combined_array = np.hstack([df.to_numpy() for df in padded_dfs])

        combined_traces = pd.DataFrame(
            combined_array,
            columns=combined_reg.index,
            index=unified_index
        )

        return cls(combined_reg, combined_traces, copy=False)

    @functools.wraps(pd.DataFrame.reset_index)
    def reset_index(self, *args, drop=True, **kwargs):

        reg = self.reg.reset_index(*args, drop=drop, **kwargs)

        traces = self.traces.copy()
        traces.columns = reg.index

        return Traces(reg, traces)

    @functools.wraps(pd.DataFrame.set_index)
    def set_index(self, *args, **kwargs):

        reg = self.reg.set_index(*args, **kwargs)
        assert reg.index.is_unique

        traces = self.traces.copy()
        traces.columns = reg.index

        return Traces(reg, traces)

    @functools.wraps(pd.DataFrame.__eq__)
    def __eq__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__eq__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__ne__)
    def __ne__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__ne__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__ge__)
    def __ge__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__ge__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__gt__)
    def __gt__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__gt__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__le__)
    def __le__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__le__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__lt__)
    def __lt__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__lt__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__add__)
    def __add__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__add__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__radd__)
    def __radd__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__radd__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__sub__)
    def __sub__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__sub__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__rsub__)
    def __rsub__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__rsub__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__mul__)
    def __mul__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__mul__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__rmul__)
    def __rmul__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__rmul__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__truediv__)
    def __truediv__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__truediv__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__rtruediv__)
    def __rtruediv__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__rtruediv__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__floordiv__)
    def __floordiv__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__floordiv__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__rfloordiv__)
    def __rfloordiv__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__rfloordiv__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__neg__)
    def __neg__(self):
        return self.replace_traces(
            self.traces.__neg__(),
        )

    @functools.wraps(pd.DataFrame.__neg__)
    def __mod__(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.__mod__(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.__array__)
    def __array__(self, *args, **kwargs):
        return self.traces.__array__(*args, **kwargs)

    @functools.wraps(pd.DataFrame.__array_ufunc__)
    def __array_ufunc__(self, *args, **kwargs):
        return self.traces.__array_ufunc__(*args, **kwargs)

    def get(self, idx=None) -> pd.Series:
        """return a single trace. If no index it's given, we assume there is only one"""
        if idx is None:
            assert len(self.traces.columns) == 1, f'Found too many traces:\n{self.reg}'
            idx = self.index[0]

        # noinspection PyTypeChecker
        return self.traces.loc[:, idx]

    def get_df(self, col, expect_unique=True) -> pd.DataFrame:
        """
        return the traces with the columns changed to be a given property
        """
        traces = self.traces

        new_col = self.reg[col]
        assert not expect_unique or new_col.is_unique
        traces = traces.T.set_index(new_col).T

        return traces

    def groupby_mix(self, by, how):

        if not isinstance(by, list):
            by = [by]

        # if drop is None:
        #     # noinspection PyUnresolvedReferences
        #     different = (self.reg.groupby(by).nunique() > 1).any()
        #     drop = different.index[different]

        by = [self.reg[col] for col in by]

        agg_traces = {}
        agg_reg = {}

        for k, straces in self.traces.T.groupby(by):
            straces = straces.T
            sreg = self.reg.loc[straces.columns]

            same_values = sreg.nunique(dropna=True) == 1
            none_missing = sreg.isna().sum() == 0
            sreg = sreg.loc[:, same_values & none_missing].drop_duplicates()
            assert len(sreg) == 1, f'Expected single trace for {k}. Got {len(sreg)}: {sreg.nunique()}'
            idx = sreg.index[0]

            agg_reg[idx] = sreg.iloc[0]
            # noinspection PyTypeChecker
            agg_traces[idx] = how(straces, axis=1)

        agg_reg = pd.DataFrame.from_dict(agg_reg, orient='index')
        agg_reg.sort_index(inplace=True)

        agg_traces = pd.DataFrame(agg_traces)
        agg_traces.sort_index(inplace=True, axis=1)

        return Traces(
            reg=agg_reg,
            traces=agg_traces,
        )

    def groupby_mean(self, by):
        return self.groupby_mix(by, how=pd.DataFrame.mean)

    def groupby_median(self, by):
        return self.groupby_mix(by, how=pd.DataFrame.median)

    def groupby_std(self, by):
        return self.groupby_mix(by, how=pd.DataFrame.std)

    def groupby_max(self, by):
        return self.groupby_mix(by, how=pd.DataFrame.max)

    def histograms(self, bins=None, density=None, weights=None) -> pd.DataFrame:
        if bins is None:
            bins = 100

        if isinstance(bins, int):
            bins = np.linspace(
                np.nanmin(self.traces.values),
                np.nanmax(self.traces.values),
                bins + 1,
            )

        df = pd.DataFrame({
            k: np.histogram(trace, bins=bins, density=density, weights=weights)[0]
            for k, trace in self.items()
        })

        df.index = pd.IntervalIndex.from_breaks(bins)

        return df

    def items(self, col=None, *, pbar=None):
        """
        Returns an iterator to go over each trace.
        If 'col' is None, then the key will be the index.
        If it is not, then the key will be the value of the corresponding column
        (which must be unique).

            for exp_name, trace in beta.items('exp_name', pbar=True):
                pass

        """
        if col is not None:
            data = self.get_df(col)
        else:
            data = self.traces

        return _optional_pbar(data.items(), total=len(data.columns), pbar=pbar)

    def histograms2d(self, vbins=None, tbins=None, rolling_win=None, pbar=None):
        """
        Extracts a 2D histogram for each trace where the first dimension is the time and the second the value.

        This can be useful to look at how the distribution of values changes over time.

        All histograms will have the same bins.
        """

        if vbins is None:
            vbins = 100

        if isinstance(vbins, int):
            vbins = np.linspace(
                np.nanmin(self.traces.values),
                np.nanmax(self.traces.values),
                vbins,
            )

        if tbins is None:
            tbins = float(self.sampling_period)

        if isinstance(tbins, float):
            tbins = self.get_global_win().arange(tbins)

        hists = {}
        for k, trace in self.items(pbar=pbar):
            trace = trace.dropna()

            h, t_edges, v_edges = np.histogram2d(
                trace.index,
                trace.values,
                bins=(tbins, vbins),
            )

            h = pd.DataFrame(
                h,
                index=pd.IntervalIndex.from_breaks(t_edges),
                columns=pd.IntervalIndex.from_breaks(v_edges),
            )

            if rolling_win is not None:
                h = h.rolling(rolling_win, center=True).mean()

            hists[k] = h

        hists = dd.DataDict(self.reg, hists)

        return hists

    def normalize_by_quantiles(self, qmin=0.05, qmax=.95, win=None):

        if win is not None:
            traces = self.crop(win)
            assert len(traces.time) > 0, f'No data in {win}'

        else:
            traces = self

        vmin = traces.traces.quantile(qmin)
        vmax = traces.traces.quantile(qmax)

        return (self - vmin) / (vmax - vmin)

    def iter_grouped(self, groupby, pbar=None):

        grouped = self.reg.groupby(groupby, sort=False)

        for k, sub_reg in _optional_pbar(grouped, total=len(grouped.groups), pbar=pbar):
            sub_traces = Traces(
                reg=sub_reg,
                traces=self.traces.loc[:, sub_reg.index],
            )

            yield k, sub_traces

    @functools.wraps(pd.DataFrame.clip)
    def clip(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.clip(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.replace)
    def replace(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.replace(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.fillna)
    def fillna(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.fillna(*args, **kwargs),
        )

    @functools.wraps(pd.DataFrame.abs)
    def abs(self):
        return self.replace_traces(
            self.traces.abs(),
        )

    def unique_sorted(self, col, order=None):
        """
        Returns unique values of a column with a particular preference for some of them in their ordering.
        For example, if
            vals = ['x', 'y', 'a', 'c']

        and
            order = ['a', 'b', 'c']

        this will return

            sorted_vals = ['a', 'c', 'x', 'y']

        This is useful when we want to iterate over the values for plotting and we know the general order,
        but not the particular case, so we are as consistent as possible across plot.

        """
        order = order or []

        vals = self[col].unique()

        order_idcs = {key: idx for idx, key in enumerate(order)}

        sorted_vals = sorted(vals, key=lambda x: order_idcs.get(x, float('inf')))

        return sorted_vals

    def log10(self, drop_inf=False):
        log = self.apply(np.log10)

        if drop_inf:
            log.traces.replace(-np.inf, np.nan, inplace=True)
            log.traces.replace(+np.inf, np.nan, inplace=True)

        return log

    def power10(self):
        """opposite of log10"""
        return self.apply(lambda x: np.power(10, x))

    def square(self):
        return self.apply(np.square)

    def sqrt(self):
        return self.apply(np.sqrt)

    def unwrap(self, period=1, axis=0):
        if np.any(np.isnan(self.traces)):
            logging.warning(f'Unwrapping does not support nans')

        return self.replace_traces(
            np.unwrap(self.traces.values, period=period, axis=axis)
        )

    def crop(
            self,
            win: timeslice.Win,
            **kwargs,
    ):
        win = timeslice.Win(*win)
        new = self.apply(lambda trace: win.crop_df(trace, **kwargs))
        return new

    def crop_centered(
            self,
            duration,
            **kwargs,
    ):
        win = timeslice.Win.build_centered(np.mean(self.time), duration)
        return self.crop(win, **kwargs)

    def shift_time(
            self,
            ref_time: float | pd.Series | np.ndarray,
    ):
        if isinstance(ref_time, (pd.Series, np.ndarray)):
            return self.shift_time_each(
                ref_time
            )
        else:
            return self.replace_traces(
                self.traces.set_index(self.time + ref_time)
            )

    def shift_time_each(
            self,
            shifts: pd.Series,
            neg=False,
    ):
        """
        Shift each individual trace by the corresponding shift.
        Useful if the same event happened at a different time each.
        Note the resulting traces may not align in their sampling rate, depending
        on whether the shifts align. Use 'resample' to re-align the traces.
        :param shifts:
            The shifts to apply to each individual trace.
            If a string, it is assumed to be the name of the column in the registry.
        :param neg:
            To negate the shifts, use neg=True. Useful if shifts is a column.
        :return:
        """
        if isinstance(shifts, str):
            shifts = self[shifts]

        if isinstance(shifts, np.ndarray):
            shifts = pd.Series(shifts, index=self.index)

        assert isinstance(shifts, pd.Series)
        shifts = shifts.reindex(self.index)
        assert shifts.notna().all()

        if neg:
            shifts = shifts * -1

        shifted_traces = {
            k: pd.Series(trace.values, index=trace.index + shifts[k])
            for k, trace in self.items()
        }

        # TODO
        # This might be slow (due to re-indexing) and generate many np.nans if the shifts dont align
        # which can be common with floating errors.
        # This can be solved post-hoc with 'resample', but we could avoid it all together
        # by directly resampling here in the same way that 'from_dict_resampled' works.
        shifted_traces = pd.DataFrame(shifted_traces)

        return self.replace_traces(
            shifted_traces
        )

    def cross_corr(self, pairs, pearson=True, mode="same"):
        """
        Compute cross-correlations between selected pairs of traces.

        Parameters
        ----------
        pairs : iterable of tuple[int, int]
            List of (i, j) index pairs referring to traces in `self`.
        pearson : bool, default True
            If True, compute Pearson-normalized cross-correlation.
        mode : {'same', 'full', 'valid'}
            Correlation mode, passed to scipy.signal.correlate.

        Returns
        -------
        dict[tuple[int, int], pd.Series]
            Mapping (i, j) -> cross-correlation series indexed by lag.
        """

        def _single_xcorr(trace_i, trace_j):
            # drop NaNs independently, then align on common time index
            ti = trace_i.dropna()
            tj = trace_j.dropna()

            if len(ti) == 0 or len(tj) == 0:
                return pd.Series(dtype=float)

            # align on common index
            ti, tj = ti.align(tj, join="inner")

            if len(ti) == 0:
                return pd.Series(dtype=float)

            if pearson:
                ti = ti - ti.mean()
                tj = tj - tj.mean()

            t = ti.index - ti.index.min()

            if mode == "same":
                lags = t - t[len(t) // 2]
            elif mode == "full":
                lags = np.concatenate([
                    t[::-1] * -1,
                    t[1:],
                ])
            else:
                assert mode == "valid"
                lags = [0]

            xcorr = scipy.signal.correlate(
                ti.values,
                tj.values,
                mode=mode,
            )

            xcorr = pd.Series(xcorr, index=lags)

            if pearson:
                denom = np.sqrt(ti.var() * tj.var()) * len(ti)
                if denom != 0:
                    xcorr = xcorr / denom
                else:
                    xcorr[:] = np.nan

            return xcorr

        out = {}
        for k, i, j in pairs[['first', 'second']].itertuples():
            trace_i = self.traces[i]
            trace_j = self.traces[j]
            out[k] = _single_xcorr(trace_i, trace_j)

        return self.__class__(
            pairs,
            pd.DataFrame(out),
        )

    def cross_corr_rolling_by(
        self,
        pair_by,
        sort_by=None,
        **kwargs,
    ):
        if sort_by is None:
            sort_by = pair_by

        def sort_pair(a, b):
            if self.reg.loc[a, sort_by] < self.reg.loc[b, sort_by]:
                return a, b
            else:
                return b, a

        groups = {
            pair_idx: sort_pair(a, b)
            for pair_idx, (a, b) in self.reg.groupby(pair_by).groups.items()
        }

        pairs = pd.DataFrame.from_dict(groups, orient='index', columns=['first', 'second'])

        xcorrs = self.cross_corr_rolling(
            pairs=pairs,
            **kwargs,
        )

        return pairs, xcorrs

    def cross_corr_rolling(
            self,
            pairs: pd.DataFrame,
            lags_ms: np.ndarray,
            sliding_win_ms: float,
            sliding_step_ms: float = None,
            pbar=None,
            pbar_single=None,
            pearson=True,
    ):
        sampling_period = self.sampling_period

        if sliding_step_ms is None:
            sliding_step_ms = sampling_period

        if isinstance(lags_ms, tuple) and len(lags_ms) == 2:
            lags_ms = np.arange(*lags_ms, sampling_period)

        if isinstance(pairs, list):
            pairs = pd.DataFrame([(0, 1)], columns=['first', 'second'])

        to_iter = pairs[['first', 'second']].itertuples()

        xcorrs = {}

        for i, k0, k1 in _optional_pbar(to_iter, total=len(pairs), pbar=pbar):
            xcorr = _rolling_cross_corr_ms(
                self.traces[k0].values,
                self.traces[k1].values,
                sampling_period_ms=sampling_period,
                lags_ms=lags_ms,
                sliding_win_ms=sliding_win_ms,
                sliding_step_ms=sliding_step_ms,
                pbar=pbar_single,
                pearson=pearson,
            )

            win_centers = (
                    np.arange(xcorr.shape[1]) * sliding_step_ms
                    + self.traces.index.min()
                    + sliding_win_ms * .5
            )

            xcorrs[i] = pd.DataFrame(
                xcorr.T,
                index=win_centers,
                columns=lags_ms,
            )

        return xcorrs

    def corr_rolling(
            self,
            template: np.ndarray,
            lags_ms: np.ndarray,
            sliding_win_ms: float,
            sliding_step_ms: float = None,
            pbar=None,
            pbar_single=None,
            pearson=True,
    ):
        sampling_period = self.sampling_period

        if sliding_step_ms is None:
            sliding_step_ms = sampling_period

        if isinstance(lags_ms, tuple) and len(lags_ms) == 2:
            lags_ms = np.arange(*lags_ms, sampling_period)

        xcorrs = {}

        if isinstance(template, pd.Series):
            template = template.values

        for k in _optional_pbar(self.index, total=len(self.index), pbar=pbar):
            xcorr = _rolling_cross_corr_ms(
                self.traces[k].values,
                template,
                sampling_period_ms=sampling_period,
                lags_ms=lags_ms,
                sliding_win_ms=sliding_win_ms,
                sliding_step_ms=sliding_step_ms,
                pbar=pbar_single,
                pearson=pearson,
            )

            win_centers = (
                    np.arange(xcorr.shape[1]) * sliding_step_ms
                    + self.traces.index.min()
                    + sliding_win_ms * .5
            )

            xcorrs[k] = pd.DataFrame(
                xcorr.T,
                index=win_centers,
                columns=lags_ms,
            )

        return xcorrs

    def auto_corr_rolling(
            self,
            lags_ms: np.ndarray,
            sliding_win_ms: float,
            sliding_step_ms: float = None,
            pbar=None,
            pbar_each=None,
            pearson=True,
            key=None,
    ):

        if key is not None:
            assert self[key].is_unique

        sampling_period = self.sampling_period

        if sliding_step_ms is None:
            sliding_step_ms = sampling_period

        if isinstance(lags_ms, tuple) and len(lags_ms) == 2:
            lags_ms = np.arange(*lags_ms, sampling_period)

        acorrs = {}

        for k, trace in self.items(pbar=pbar):
            acorr = _rolling_cross_corr_ms(
                trace.values,
                trace.values,
                sampling_period_ms=sampling_period,
                lags_ms=lags_ms,
                sliding_win_ms=sliding_win_ms,
                sliding_step_ms=sliding_step_ms,
                pbar=pbar_each,
                pearson=pearson,
            )

            win_centers = (
                    np.arange(acorr.shape[1]) * sliding_step_ms
                    + self.traces.index.min()
                    + sliding_win_ms * .5
            )

            acorr = pd.DataFrame(
                acorr.T,
                index=win_centers,
                columns=lags_ms,
            )

            acorrs[k] = acorr

        if key is not None:
            acorrs = {
                self.loc[k, key]: v
                for k, v in acorrs.items()
            }

        return acorrs

    def auto_corr(self, pearson=True, mode='same'):

        def _single_acorr(trace):

            # note that the presence of nans means different
            # trace may have effective different lengths
            # which means the normalization (where we divide by the length)
            # needs to be done per trace
            trace = trace.dropna()

            if len(trace) == 0:
                return trace

            if pearson:
                trace = trace - trace.mean()

            t = trace.index - trace.index.min()

            if mode == 'same':
                lags = t - t[len(t) // 2]

            elif mode == 'full':
                lags = np.concatenate([
                    t[::-1] * -1,
                    t[1:],
                ])
            else:
                assert mode == 'valid'
                lags = [0]

            # noinspection PyUnresolvedReferences
            acorr = scipy.signal.correlate(
                trace,
                trace,
                mode=mode,
            )

            acorr = pd.Series(acorr, index=lags)

            if pearson:
                acorr = acorr / (trace.var() * len(trace))

            return acorr

        return self.apply(_single_acorr)

    @staticmethod
    def _match_traces_wins(reg: pd.DataFrame, windows, **kwargs):

        reg: pd.DataFrame = reg.copy()
        # reg.drop(['ref'], axis=1, inplace=True)

        if reg.index.name is None:
            reg.index.name = 'index_reg'

        reg_index_name = reg.index.name
        reg.reset_index(inplace=True)

        wins = windows.wins.copy()

        if wins.index.name is None or wins.index.name == reg_index_name:
            wins.index.name = 'index_wins'

        wins.reset_index(inplace=True)

        merged = pd.merge(
            wins,
            reg,
            how='left',
            **kwargs,
        )

        merged.dropna(subset=reg_index_name, inplace=True)

        return merged, reg_index_name

    def extract_all(  #TODO rewrite
            self,
            windows: timeslice.Windows,
            upsampling_ms=None,
            pbar=None,
    ):
        """
        Extract all traces using a set of time windows.

        This method applies each window to all traces, generating a new collection of traces
        where each original trace is split into multiple segments corresponding to the given windows.

        Parameters
        ----------
        windows : Windows
            A `Windows` object defining the time ranges for extraction.
        upsampling_ms : float, optional
            If provided, upsample traces to this time resolution before extraction.
        pbar : bool or tqdm-like, optional
            Whether to display a progress bar.

        Returns
        -------
        Traces
            A new `Traces` object where each segment corresponds to a window.

        Notes
        -----
        - This method differs from `extract()`, which maps traces to their corresponding windows.
        - If windows overlap, extracted traces may have duplicated time segments.
        """

        if upsampling_ms is None:
            upsampling_ms = self.sampling_period

        interp_traces = windows.interp_df(
            self.traces,
            step=upsampling_ms,
            pbar=pbar,
        )

        traces = pd.concat(interp_traces, axis=1, names=[windows.index.name])

        new = traces.columns.to_frame(index=False)

        new.rename(columns={new.columns[-1]: f'precut_{new.columns[-1]}'}, inplace=True)

        wins_reg = windows.wins.reindex(new.iloc[:, 0])
        wins_reg.index = new.index

        traces_reg = self.reg  # .drop(['ref'], axis=1)
        traces_reg = traces_reg.reindex(new.iloc[:, 1])
        traces_reg.index = new.index

        reg = pd.concat([wins_reg, traces_reg, new], axis=1)

        dups = reg.columns.duplicated()
        if np.any(dups):
            logging.warning(
                f'Dropping duplicated columns: ' + ', '.join(list(reg.columns[dups]))
                + '. Maybe you want cut_merge?'
            )
            reg = reg.loc[:, ~dups]

        traces.columns = reg.index

        return self.from_df(
            traces=traces,
            reg=reg,
        )

    def extract(self, wins, align=None, upsampling_ms=None):
        """
        Extract segments from traces using a `Windows` object.

        This method selects and extracts trace segments according to the time windows
        defined in `windows`, ensuring that each trace is matched with its corresponding
        windows. It is the primary method for time-based trace extraction.

        Parameters:
        -----------
        wins : Windows
            A `Windows` object defining the start and stop times for extraction.

        **kwargs :
            Additional arguments passed to fine-tune extraction behavior through pd.merge.

        Returns:
        --------
        Traces
            A new `Traces` object containing only the extracted segments.
        """

        upsampling_ms = upsampling_ms or self.sampling_period * 2

        matched_reg = Traces.match(self, wins, left_ref='tr_idx', right_ref='win_idx')

        if len(matched_reg) == 0:
            logging.error('No matches found')

        multi_cut = []

        for win_idx, sel in matched_reg.groupby('win_idx', sort=False):

            sel_traces = self.traces.loc[:, sel['tr_idx'].values]

            assert not np.any(np.isnan(sel_traces.index))

            win: timeslice.Win = wins.get(win_idx)

            cut_traces = win.interp_df(sel_traces, step=upsampling_ms)
            cut_traces.columns = sel.index

            if align is not None:
                ref = wins.relative_time(align).loc[win_idx]
                cut_traces.index = cut_traces.index - ref

            assert not np.any(np.isnan(cut_traces.index))

            multi_cut.append(cut_traces)

        multi_cut_traces = pd.concat(multi_cut, axis=1)

        # need to resort because the group-by above may process traces out of order
        matched_reg.sort_index(inplace=True)
        multi_cut_traces.sort_index(axis=1, inplace=True)

        result = Traces.from_df(reg=matched_reg, traces=multi_cut_traces)

        return result

    def copy(self):
        return self.__class__(
            reg=self.reg.copy(),
            traces=self.traces.copy(),
        )

    def set_cols(self, extra: pd.DataFrame, *, suffix=None, prefix=None):
        """add extra columns describing properties of these windows"""
        assert len(extra) == len(self.index)
        new = self.copy()

        if suffix is not None:
            extra = extra.add_suffix(suffix)

        if prefix is not None:
            extra = extra.add_prefix(prefix)

        for col, vals in extra.items():
            new[col] = vals

        return new

    @functools.wraps(pd.merge)
    def merge_reg(self, extra: pd.DataFrame, **kwargs):
        return self.__class__(
            pd.merge(
                self.reg,
                extra,
                **kwargs,
            ),
            self.traces,
        )

    @functools.wraps(pd.DataFrame.drop)
    def drop(self, *args, **kwargs):
        return self.__class__(
            self.reg.drop(*args, **kwargs),
            self.traces,
        )

    def _apply_mask(self, mask) -> Self:
        return self.__class__(
            reg=self.reg.loc[mask],
            traces=self.traces.loc[:, mask],
        )

    def _replace_reg(self, reg) -> Self:
        return self.__class__(
            reg=reg,
            traces=self.traces,
        )

    @property
    def shape(self):
        return self.reg.shape

    @property
    def values(self):
        return self.traces.values

    @property
    def time(self):
        return self.traces.index

    @property
    def tloc(self):
        return self.traces.loc

    def lookup(self, times, interp=True) -> pd.Series:
        """
        Look up a different time for each trace.
        For example, imagine these traces are time series of different animals
        and each one of them has a different cycle-duration.

        :param times: A series of times with index equal to this traces index.
            Alternatively, a str identifying a column to lookup.

        :param interp: Whether to interpolate the traces to look up the given
            times if they don't perfectly align with our sampling
        """

        if isinstance(times, str):
            times = self[times]

        if isinstance(times, (np.ndarray, pd.Index)):
            times = pd.Series(times, index=self.index)

        if np.isscalar(times):
            times = pd.Series(times, index=self.index)

        def lookup_single(s, t) -> float:
            if interp:
                return np.interp(t, s.index, s.values).item()

            else:
                return s.loc[t]

        return pd.Series({
            k: lookup_single(self.get(k), t)
            for k, t in times.items()
        })

    def apply(self, *args, **kwargs):
        new = self.traces.apply(*args, **kwargs)
        return self.replace_traces(new)

    @functools.wraps(pd.DataFrame.applymap)
    def applymap(self, func):
        to_map = func
        if isinstance(func, dict):
            to_map = lambda x: func[x]

        mapped = self.traces.applymap(to_map)

        return self.replace_traces(mapped)

    def replace_traces(self, others: dict | np.ndarray | pd.DataFrame):

        if isinstance(others, dict):
            others: pd.DataFrame = pd.DataFrame(others)

        if not isinstance(others, pd.DataFrame):
            assert others.shape == self.traces.shape
            others: pd.DataFrame = pd.DataFrame(
                others,
                index=self.traces.index,
                columns=self.traces.columns,
            )

        missing = others.columns.difference(self.reg.index)
        if len(missing) > 0:
            logging.warning(f'Missing reg entries for {len(missing)} traces')

        common = others.columns.intersection(self.reg.index)

        return self.__class__(
            reg=self.reg.loc[common],
            traces=others.loc[:, common],
        )

    def sort_values(self, *args, **kwargs):
        reg = self.reg.sort_values(*args, **kwargs)

        return self.__class__(
            reg=reg,
            traces=self.traces.reindex(reg.index, axis=1),
        )

    def sort_index(self, *args, **kwargs):
        reg = self.reg.sort_index(*args, **kwargs)

        return self.__class__(
            reg=reg,
            traces=self.traces.reindex(reg.index, axis=1),
        )

    def contains_nan(self):
        return bool(np.any(
            self.traces.isna().values
        ))

    def drop_missing(self, how='any'):
        """Drop timepoints with missing data for any trace"""
        # noinspection PyTypeChecker
        traces = self.traces.dropna(axis=0, how=how)
        return self.replace_traces(traces)

    def drop_empty(self, how='all'):
        """Drop entire traces if they are completely missing data"""
        # noinspection PyTypeChecker
        traces = self.traces.dropna(axis=1, how=how)
        return self.replace_traces(traces)

    def drop_tight(self):
        """
        Drop leading and trailing rows that are entirely NaN.
        Interior all-NaN rows are preserved.
        """
        mask = self.traces.notna().any(axis=1)

        if not mask.any():
            sel = self.traces.iloc[0:0]
            return self.replace_traces(sel)

        first = mask.idxmax()
        last = mask[::-1].idxmax()

        sel = self.traces.loc[first:last]
        return self.replace_traces(sel)

    def get_global_win(self) -> timeslice.Win:
        return timeslice.Win(
            self.time.min(),
            self.time.max(),
        )

    @property
    def sampling_period(self) -> float:
        return _estimate_sampling_period(self.time)

    @property
    def sampling_rate(self) -> float:
        sampling_rate = 1. / (self.sampling_period * timeslice.MS_TO_S)

        if sampling_rate.is_integer():
            sampling_rate = int(sampling_rate)

        return sampling_rate

    def gradient(self):
        sampling_period = self.sampling_period
        return self.apply(
            lambda trace: np.gradient(trace, sampling_period)
        )

    def diff(self):
        return self.apply(pd.Series.diff)

    @functools.wraps(pd.DataFrame.max)
    def max(self, *args, **kwargs):
        return self.traces.max(*args, **kwargs)

    @functools.wraps(pd.DataFrame.min)
    def min(self, *args, **kwargs):
        return self.traces.min(*args, **kwargs)

    @functools.wraps(pd.DataFrame.idxmax)
    def idxmax(self, *args, **kwargs):
        return self.traces.idxmax(*args, **kwargs)

    @functools.wraps(pd.DataFrame.idxmin)
    def idxmin(self, *args, **kwargs):
        return self.traces.idxmin(*args, **kwargs)

    @functools.wraps(pd.DataFrame.mean)
    def mean(self, *args, **kwargs):
        return self.traces.mean(*args, **kwargs)

    def cumsum(self, *args, **kwargs):
        return self.replace_traces(
            self.traces.cumsum(*args, **kwargs)
        )

    def mean_rolling(self, *args, center=True, min_periods=1, **kwargs):
        rolling = self.traces.rolling(*args, center=center, min_periods=min_periods, **kwargs)

        return self.replace_traces(
            rolling.mean(),
        )

    def mean_rolling_gaussian(self, std_ms, center=True, min_periods=None):  # TODO inconsistent window def

        std_idcs = std_ms / self.sampling_period

        traces = self.traces.rolling(
            int(std_idcs * 4),
            win_type='gaussian',
            center=center,
            min_periods=min_periods,
        ).mean(std=std_idcs)

        return self.replace_traces(traces)

    def sum_rolling(self, *args, center=True, min_periods=1, **kwargs):
        rolling = self.traces.rolling(*args, center=center, min_periods=min_periods, **kwargs)

        return self.replace_traces(
            rolling.sum(),
        )

    def median_rolling(self, *args, center=True, min_periods=1, **kwargs):
        rolling = self.traces.rolling(*args, center=center, min_periods=min_periods, **kwargs)

        return self.replace_traces(
            rolling.median(),
        )

    def std_rolling(self, *args, center=True, min_periods=1, **kwargs):
        rolling = self.traces.rolling(*args, center=center, min_periods=min_periods, **kwargs)

        return self.replace_traces(
            rolling.std(),
        )

    def zscore_rolling(self, sliding_win_ms):
        win_size = sliding_win_ms / self.sampling_period

        assert win_size == int(win_size)

        win_size = int(win_size)
        assert win_size > 0

        mean = self.mean_rolling(win_size)
        std = self.std_rolling(win_size)

        return (self - mean.traces) / std.traces

    @functools.wraps(pd.DataFrame.median)
    def median(self, *args, **kwargs):
        return self.traces.median(*args, **kwargs)

    @functools.wraps(pd.DataFrame.quantile)
    def quantile(self, *args, **kwargs):
        return self.traces.quantile(*args, **kwargs)

    @functools.wraps(pd.DataFrame.std)
    def std(self, *args, **kwargs):
        return self.traces.std(*args, **kwargs)

    @functools.wraps(pd.DataFrame.var)
    def var(self, *args, **kwargs):
        return self.traces.var(*args, **kwargs)

    @functools.wraps(pd.DataFrame.sum)
    def sum(self, *args, **kwargs):
        return self.traces.sum(*args, **kwargs)

    def zscore(self):
        return (self - self.mean()) / self.std()

    def resample(self, period, start=None, stop=None):

        if start is None:
            start = self.time.min()

        if stop is None:
            stop = self.time.max()

        win = timeslice.Win(start, stop)

        return self.__class__.from_df(
            reg=self.reg,
            traces=win.interp_df(self.traces, step=period),
        )

    def downsample_factor(self, factor, offset=None):
        return self.__class__(
            reg=self.reg,
            traces=self.traces.iloc[offset::factor],
        )

    def are_continuously_sampled(self, atol=1.e-6) -> bool:
        """Check that there are no gaps in the sampling time"""
        dts = np.diff(self.time)
        return np.allclose(dts[0], dts, atol=atol)

    def downsample(self, period):
        current = self.sampling_period

        assert np.isclose(period % current, 0), \
            f'New period ({period}) must be a multiple of current period ({current})'

        factor = int(period / current)

        return self.downsample_factor(factor)

    def interp(self, times: pd.Series):

        times = np.asarray(times)

        interpolated = np.column_stack([
            np.interp(times, self.time, self.traces[col].values)
            for col in self.traces.columns
        ])

        new_traces = pd.DataFrame(
            interpolated,
            index=times,
            columns=self.traces.columns,
        )

        return self.replace_traces(new_traces)

    def filter_pass(self, hz: tuple, **kwargs):
        """
        A combined call to low_pass / high_pass / band_pass.

        This is useful to quickly switch the filtering in an analysis
        with just one parameter.

        :param hz: Must be a tuple (or None) defining a Hz range for a band pass filter.
        If one of the ends is inf, it turns into a high or low pass filter:

            Low pass:
                (-np.inf, 5)
                (None, 5)

            Band pass:
                (20, 50)

            high pass:
                (50, np.inf)
                (50, None)

            No filter:
                None
                (-np.inf, np.inf)

        """
        if hz is None:
            return self

        low, high = hz
        low_open = low is None or np.isclose(low, 0) or np.isinf(low) or np.isnan(low)
        high_open = high is None or np.isclose(high, 0) or np.isinf(high) or np.isnan(high)

        if low_open and not high_open:
            return self.low_pass(high, **kwargs)

        elif not low_open and high_open:
            return self.high_pass(low, **kwargs)

        elif not low_open and not high_open:
            return self.band_pass(low, high, **kwargs)

        else:
            return self

    def filtfilt(self, *params):
        """a version that tolerates nans at the start/end of each col"""
        padded = self.traces.ffill().bfill().values
        filtered_data = scipy.signal.filtfilt(*params, padded, axis=0)
        filtered_data[np.isnan(self.traces.values)] = np.nan

        new_traces = pd.DataFrame(
            filtered_data,
            index=self.traces.index,
            columns=self.traces.columns,
        )
        return self.replace_traces(new_traces)

    def band_pass(self, low_hz, high_hz, *, order=2):
        assert self.are_continuously_sampled()

        sampling_hz = self.sampling_rate

        nyquist_freq = sampling_hz / 2
        assert high_hz <= nyquist_freq, \
            f'High Hz ({high_hz}) must be <= Nyquist ({nyquist_freq})'

        low = low_hz / nyquist_freq
        high = high_hz / nyquist_freq

        # noinspection PyUnresolvedReferences
        params = scipy.signal.butter(order, [low, high], btype='band')

        return self.filtfilt(*params)

    def low_pass(self, high_hz, *, order=2):
        assert self.are_continuously_sampled()

        sampling_hz = self.sampling_rate

        nyquist_freq = sampling_hz / 2
        assert high_hz <= nyquist_freq, \
            f'High Hz ({high_hz}) must be <= Nyquist ({nyquist_freq})'

        high = high_hz / nyquist_freq

        # noinspection PyUnresolvedReferences
        params = scipy.signal.butter(order, high, btype='low')

        return self.filtfilt(*params)

    def high_pass(self, low_hz, *, order=2):
        assert self.are_continuously_sampled()

        sampling_hz = self.sampling_rate

        nyquist_freq = sampling_hz / 2
        low = low_hz / nyquist_freq

        # noinspection PyUnresolvedReferences
        params = scipy.signal.butter(order, low, btype='high')

        return self.filtfilt(*params)

    def _spectral_analysis_single(self, k, spec_func, take_abs, **kwargs):
        """Apply a spectral analysis function to a single trace"""

        assert self.are_continuously_sampled()

        sampling_rate = self.sampling_rate

        data = self.traces[k]

        f_stft, t_stft, z_xx = spec_func(data, fs=sampling_rate, **kwargs)

        time_offset = self.time.min()

        time = t_stft * timeslice.ms(seconds=1) + time_offset

        spec = z_xx.T

        if take_abs:
            spec = np.abs(spec)

        df = pd.DataFrame(
            spec,
            columns=f_stft,
            index=time,
        )

        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)

        return df.rename_axis(index='time', columns='freq')

    def _spectral_analysis(self, spec_func, pbar=tqdm, take_abs=True, db=True, **kwargs):
        """
        Apply a spectral analysis function to each trace

        spec_func must return the same as scipy.signal.spectrogram, that is:
        freqs (M array), time (N array), spec (MxN array)
        """
        res = {}

        for k in _optional_pbar(self.index, total=len(self.index), pbar=pbar):
            spec = self._spectral_analysis_single(k, spec_func, take_abs=take_abs, **kwargs)

            if db:
                spec = 10 * np.log10(spec)

            res[k] = spec

        return res

    def spectral_analysis_stft(self, nperseg=256, noverlap=128, **kwargs):
        """Short-Time Fourier Transform (STFT)"""
        return self._spectral_analysis(
            scipy.signal.stft,
            nperseg=nperseg,
            noverlap=noverlap,
            **kwargs,
        )

    def spectral_analysis_wavelet(self, freqs=None, wavelet='morl', **kwargs):
        """Wavelet Transform (Morlet wavelet)"""
        import pywt
        sampling_rate = self.sampling_rate

        if freqs is None:
            freqs = np.geomspace(1, 100, 101)  # np.arange(1, 128)

        center_frequency = 0.84  # Morlet wavelet typical center frequency
        scales = center_frequency / (freqs / sampling_rate)

        def wavelet_single(data, fs, **wv_kwargs):
            coef, freqs_wt = pywt.cwt(data, sampling_period=1. / fs, **wv_kwargs)
            freqs_wt = freqs  # convert scales to frequency
            return freqs_wt, np.arange(0, len(self.time)) / sampling_rate, coef

        return self._spectral_analysis(
            wavelet_single,
            scales=scales,
            wavelet=wavelet,
            **kwargs,
        )

    def spectrograms_overlapping(self, nperseg=256, noverlap=192, window=('tukey', 0.25), **kwargs):
        """Approximation to multi-taper spectrogram by using overlapping tukey windows"""

        return self._spectral_analysis(
            scipy.signal.spectrogram,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            **kwargs,
        )

    def spectrograms(self, segment_ms=1_000, overlap_ms=None, **kwargs):
        """Specgrograms with sensible defaults"""
        assert self.are_continuously_sampled()

        period = self.sampling_period
        nperseg = int(segment_ms / period)
        nperseg = int(np.clip(nperseg, 1, np.inf))

        if overlap_ms is None:
            overlap_ms = segment_ms * .95
        noverlap = int(overlap_ms / period)
        noverlap = int(np.clip(noverlap, 0, nperseg - 1))

        return self._spectral_analysis(
            scipy.signal.spectrogram,
            nperseg=nperseg,
            noverlap=noverlap,
            **kwargs,
        )

    def welch(
            self,
            win_len_ms=None,
            db=False,
            **kwargs,
    ):
        assert self.are_continuously_sampled()

        sampling_rate = self.sampling_rate

        power_spectral_density, sample_frequences = sleep.welch(
            self.traces.values,
            sampling_rate=sampling_rate,
            win_len_ms=win_len_ms,
            db=db,
            **kwargs,
        )

        power = pd.DataFrame(
            power_spectral_density,
            index=sample_frequences,
            columns=self.traces.columns,
        )

        return self.replace_traces(power)

    def welch_rolling(
            self,
            win_len_ms=None,
            db=False,
            sliding_len_ms=timeslice.ms(seconds=10),
            sliding_step_ms=timeslice.ms(seconds=1),
            pbar=None,
            **kwargs,
    ):
        assert self.are_continuously_sampled()
        sampling_rate = self.sampling_rate

        def welch_section(traces: pd.DataFrame):
            power_spectral_density, sample_frequences = sleep.welch(
                traces.values,
                sampling_rate=sampling_rate,
                win_len_ms=win_len_ms,
                db=db,
                **kwargs,
            )

            return sample_frequences, power_spectral_density.T

        return self.apply_rolling(
            welch_section,
            length_ms=sliding_len_ms,
            step_ms=sliding_step_ms,
            pbar=pbar,
            name='freq',
        )

    def band_power(self, bands=sleep.FREQ_BANDS, add_total=True, welch_ms=None, db=False):
        assert self.are_continuously_sampled()

        welch_ms = sleep.default_welch_ms(welch_ms, bands['freq_min'].min())

        power, bands = sleep.band_power(
            self.traces.values,
            sampling_rate=self.sampling_rate,
            bands=bands,
            welch_ms=welch_ms,
            db=db,
            add_total=add_total,
        )

        return pd.DataFrame(
            power,
            index=bands,
            columns=self.traces.columns,
        )

    def band_power_rolling(
            self,
            bands=sleep.FREQ_BANDS,
            sliding_len_ms=timeslice.ms(seconds=10),
            sliding_step_ms=timeslice.ms(seconds=1),
            db=False,
            add_total=True,
            pbar=None,
            welch_ms=None,
    ):
        """
        Extract the spectral power for each time trace using a sliding window.
        Note the windows will overlap and the value generated is assigned to its center.
        This means we cannot cover the beginning and end of the trace.
        """
        assert self.get_global_win().length >= sliding_len_ms, \
            f'Data shorter than sliding window ({self.get_global_win().length} vs {sliding_len_ms})'

        assert not self.contains_nan()

        if isinstance(bands, list):
            assert all(isinstance(b, str) for b in bands)
            bands = sleep.FREQ_BANDS.loc[bands]

        welch_ms = sleep.default_welch_ms(welch_ms, bands['freq_min'].min())

        assert sliding_len_ms >= welch_ms, \
            f'Sliding window ({sliding_len_ms} ms) must be bigger than ' \
            f'Welch window ({welch_ms}ms; lowest freq: {bands.freq_min.replace(0, np.nan).min()} Hz)'

        sampling_rate = self.sampling_rate

        def band_power_section(traces: pd.DataFrame):
            power, freqs = sleep.band_power(
                traces.values,
                sampling_rate=sampling_rate,
                bands=bands,
                welch_ms=welch_ms,
                add_total=add_total,
                db=db,
            )

            return freqs, power.T

        return self.apply_rolling(
            band_power_section,
            length_ms=sliding_len_ms,
            step_ms=sliding_step_ms,
            pbar=pbar,
            name='freq_band',
        )

    def apply_rolling(
            self,
            func,
            length_ms,
            step_ms=None,
            pbar=None,
            name='',
    ):
        """
        Extract some function for each time trace using a sliding window.
        Note the windows will overlap and the value generated is assigned to its center.
        This means we cannot cover the beginning and end of the trace.

        :param func: function to apply, should return a tuple: columns and values.
        values should be a numpy array of shape <num traces, num features>
        columns should be the labels for the features and they are assumed to be the same across
        all calls to this function (for efficicency reasons).

        :param step_ms:
            how much the sliding window is shifted on each step,
            it should be aligned with the sampling rate of the signal

        :param length_ms:
            size of the sliding window

        :param pbar:

        :param name: name of the feature extracted from the traces.

        :return: a new Traces object
        """

        if step_ms is None:
            step_ms = self.sampling_period

        # note we want indices to slice signal, which may not start at t=0
        signal_tstart, signal_tstop = self.get_global_win()
        signal_sampling_rate = self.sampling_rate
        start_off_ms = 0
        stop_off_ms = 0

        win_samples = timeslice.Windows.build_sliding_samples(
            start_ms=0 + start_off_ms,
            stop_ms=(signal_tstop - signal_tstart) + stop_off_ms,
            sampling_rate=signal_sampling_rate,
            length_ms=length_ms,
            step_ms=step_ms,
        )

        sliding_steps = win_samples.index

        starts = win_samples['start']
        stops = win_samples['stop']
        refs = win_samples['ref']

        cols = []

        results = []
        for i in _optional_pbar(sliding_steps, total=len(sliding_steps), desc='sliding win', pbar=pbar, many=100):
            section = self.traces.iloc[starts[i]:stops[i]]
            cols, result = func(section)
            results.append(result.ravel(order='C'))

        results = np.stack(results)

        results_df = pd.DataFrame(
            results,
            columns=pd.MultiIndex.from_product(
                [self.traces.columns, cols],
                names=[self.traces.columns.name, name],
            ),
            index=self.time[refs.values],
        )

        new_reg = results_df.columns.to_frame(index=False)

        merged_reg = pd.merge(
            new_reg,
            self.reg,
            how='left',
            left_on=self.traces.columns.name,
            right_index=True,
        )
        assert merged_reg.index.is_unique
        # merged_reg.drop(self.traces.columns.name, axis=1, inplace=True)

        results_df.columns = merged_reg.index

        return self.from_df(
            reg=merged_reg,
            traces=results_df,
        )
