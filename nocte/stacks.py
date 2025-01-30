"""
Functions to work with stacks of traces in the shape of tensors.

A "tensor" is just a matrix with more than 2 dimensions.
For example, a stack of spike waveforms extracted from the data for a single cell may be 3 dimensional:
    32 channels x 56 spikes x 1000 time points

Build a new stack by loading a stored one:

    from nocte import stacks as st
    stack = st.load_hdf_stack('GL881_2020803_sleep_g0/swsort/swsort_power.h5', 'power_multichan')

or by extracting it from saved data:

    import nocte.neuropixels as npix
    from nocte import stacks as st

    data = npix.DataLoader.load_spikegl(meta_path, bin_path)

    stack = Stack.extract_stack_ms(data, chan_idcs=[0, 1, 2, 3], times=[1000, 2000], win_ms=(-1000, +1000))
    

A stack can be extracted from a "raw" object via the "Stack.load*" methods.
A "raw" object must implement:
    - raw.sampling_rate
    - raw.channels
    - raw.load(slice(start, stop, load_stride), chan, adjust_gain=adjust_gain)

The Stack class wraps a xarray.DataArray object to simplify the API and
add a couple of methods. The underlying data can be accessed as stack.data.
"""
import functools
import itertools
import logging

import h5py
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
import xarray as xr
import xarray.core.coordinates
from tqdm.auto import tqdm

from nocte import timeslice
from nocte.timeslice import MS_TO_S, S_TO_MS


def _expose(parent_method):
    """
    Decorator to expose part of the xr.DataArray API
    """

    child_method = getattr(xr.DataArray, parent_method.__name__)

    @functools.wraps(child_method)
    def function(self, *args, **kwargs):

        args = [a if not isinstance(a, Stack) else a.data for a in args]
        kwargs = {k: a if not isinstance(a, Stack) else a.data for k, a in kwargs.items()}

        result = child_method(self.data, *args, **kwargs)

        if isinstance(result, xr.DataArray):

            # get rid of non-dimension coordinates because I don't
            # use datasets and they are confusing
            result = result.reset_coords(drop=True)

            return Stack(result)
        else:
            return result

    return function


class Stack:
    """
    Tensors with labels for indices and named dimensions,
    to make it easy (and efficient) to work with traces of multiple cells/channels/times

    This is implemented as a thin wrapper around xr.DataArray (inheritance is discouraged).
    The raw DataArray can be accessed through .data
    """

    def __init__(self, data: xr.DataArray):
        # make sure index name matches dim name.
        for k in data.coords.indexes.keys():
            data.coords.indexes[k].rename(k, inplace=True)

        self.data = data

    @classmethod
    def from_array(cls, values: np.ndarray, coords):
        """build from raw data"""

        if isinstance(coords, xarray.core.coordinates.Coordinates):
            # select only actual index coordinates, dropping non-dimension ones.
            # noinspection PyTypeChecker
            coords = dict(coords.indexes.items())

        coords_shape = tuple(len(v) for v in coords.values())

        if coords_shape[::-1] == values.shape:
            logging.warning(f'Transposing data')
            values = values.T

        if coords_shape != values.shape:
            raise ValueError(f'Index shape "{coords_shape}" does not match values shape {values.shape}')

        return cls(xr.DataArray(values, coords=coords, dims=list(coords.keys())))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, index_name=None, columns_name=None):
        """build from a named dataframe"""

        index_name = df.index.name if index_name is None else index_name
        if index_name is None:
            index_name = 'time' if df.index.name is None else df.index.name
        assert index_name is not None

        if columns_name is None:
            columns_name = 'traces' if df.columns.name is None else df.columns.name
        assert columns_name is not None

        df = df.rename_axis(index=index_name, columns=columns_name)

        def take_mid_interval(idx):
            if isinstance(idx, pd.IntervalIndex):
                return idx.mid
            else:
                return idx

        # noinspection PyTypeChecker
        return cls.from_array(df.values, coords={
            df.index.name: take_mid_interval(df.index),
            df.columns.name: take_mid_interval(df.columns),
        })

    @classmethod
    def from_series(cls, s: pd.Series, index_name=None):

        if index_name is None:
            index_name = s.index.name

        if index_name is None:
            index_name = 'time'

        return cls.from_array(s.values, coords={index_name: s.index})

    def __str__(self, time_dim='time'):
        coords_desc = ' x '.join([f'{len(values):,d} {name}' for name, values in self.coords.items()])
        coords_desc = '(' + coords_desc + ')'

        if time_dim in self.coords:
            time_desc = f'; sampling: {self.estimate_sampling_rate(dim=time_dim)}hz ' \
                        f'({self.estimate_sampling_period(dim=time_dim)}ms)' \
                        f'; win: {self.get_rel_win(dim=time_dim)}'

            coords_desc += f'{time_desc}'

        return f'{self.values.dtype} stack of {coords_desc}'

    def describe(self):
        print(self)

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.data._repr_html_()

    ######################################################################
    # Common operations (straight from DataArray)

    @property
    def dtype(self):
        return self.data.dtype

    @_expose
    def astype(self, *args, **kwargs):
        pass

    @_expose
    def copy(self, *args, **kwargs):
        pass

    @_expose
    def to_series(self, *args, **kwargs):
        pass

    @_expose
    def to_dataframe(self, *args, **kwargs):
        pass

    def to_dataframe2d(self):
        """
        Convert this 2d stack into a dataframe where one dimension is index
        and the other is columns.

        Xarray's to_dataframe will put all "index" coordinates into a MultiIndex.
        However, we may already have the data in the correct contiguous shape.
        This is equivalent to, but much faster than:

            data.to_series().unstack('channel')

        :return:
        """
        assert len(self.shape) == 2
        return pd.DataFrame(
            self.values,
            index=self.coords[self.dims[0]],
            columns=self.coords[self.dims[1]]
        )

    @_expose
    def reduce(self, *args, **kwargs):
        pass

    @_expose
    def dropna(self, *args, **kwargs):
        pass

    @_expose
    def std(self, *args, **kwargs):
        pass

    @_expose
    def sum(self, *args, **kwargs):
        pass

    @_expose
    def cumsum(self, *args, **kwargs):
        pass

    @_expose
    def mean(self, *args, **kwargs):
        pass

    @_expose
    def clip(self, *args, **kwargs):
        pass

    @_expose
    def diff(self, *args, **kwargs):
        pass

    # noinspection PyTypeChecker
    def gradient(self, dim='time'):
        grad = np.gradient(
            self.values,
            self.estimate_sampling_period(dim),
            axis=self.get_axis_num(dim),
        )

        return self.__class__.from_array(
            grad,
            self.coords
        )

    def apply(self, dim, func):
        """
        Apply a given function to all values along a single dimention, iterating over any other dimensions.

        This can be used, for example, if we have a function that operates in the time-domain,
        and we just want to iterate over any other dimensions (like channel).

        :param dim:
        :param func: function that takes an array and returns an array of the same size
        :returns: a stack of the same shape and coordinates.
        """

        transposed = self.transpose(..., dim)
        values_2d = transposed.values.reshape(-1, len(self.coords[dim]))

        result_2d = []

        for i, values in enumerate(values_2d):
            result_2d.append(func(values))

        result_2d = np.array(result_2d)

        # noinspection PyTypeChecker
        result_transposed = Stack.from_array(
            result_2d.reshape(transposed.values.shape),
            transposed.coords
        )

        return result_transposed.transpose(*self.coords.keys())

    def iter_dims(self, *dims, pbar=None):
        """
        Generates an iterator over the given dimensions.
        This can be used to process a stack in parts.

        For example, if we want to process ever 2D slice over 
        channel and time independently:

            for sel, substack in traces.iter_dims('channel', 'time'):
                process(substack)
                ...

        However, you need to know the exact shape of the stack to process substack.
        See iter_except for a general way to implement analysis for a known dimension.

        :param dims:
        :param pbar:

        :return: Yields pairs of <selector, Stack>.
        The selector is a dict that defines the sub stack so that:
            for selector, substack in stack.iter_dims(...):
                stack.sel(**selector) == substack
        """
        if len(dims) == 0:
            comb = dict()
            yield comb, self.sel(**comb)

        else:
            coord_value_combs = itertools.product(*[
                self.coords[name].values for name in dims
            ])

            if pbar is not None:
                coord_value_combs = pbar(list(coord_value_combs))

            for comb in coord_value_combs:
                comb = dict(zip(dims, comb))
                yield comb, self.sel(**comb)

    def iter_dim(self, dim: str, pbar=None):
        """
        Generates an iterator over a single dimension.
        For example, to process each channel independently:

            for ch, substack in traces.iter('channel'):
                process(substack)
                ...

        See iter_dims and iter_except.

        :param dim:
        :param pbar:

        :return: Yields pairs of <value, Stack>.

        The value identifies the substack in the desired dimension so that:
            for ch, substack in stack.iter_dim('channel'):
                stack.sel(channel=ch) == substack
        """
        for sel, substack in self.iter_dims(dim, pbar=pbar):
            yield sel[dim], substack

    def iter_except(self, *dims, pbar=None):
        """
        Generates an iterator over all dimensions except for the given ones.
        This can be used to process data from a single dimension, idependently of the others.

        For example, if we want to extract spectral power from the time-domain
        independently of channels or other dimensions:

            for sel, trace in traces.iter_except('time'):
                power = extract_power(trace)
                ...
        """
        return self.iter_dims(*self.get_coords_names_except(*dims), pbar=pbar)

    def apply_dataframe(self, func, new_dim, dim='time'):
        all_df = []

        for sel, trace in self.iter_except(dim):
            df = func(trace)

            for k, v in sel.items():
                df[k] = v

            all_df.append(df)

        all_df = pd.concat(all_df).rename_axis(index=new_dim)

        return all_df

    def mean_rolling(self, window, dim='time', center=True, min_periods=1, win_type='hanning'):
        """
        Apply a mean on a rolling window.
        Only supported for 2D stacks
        """

        assert self.ndim == 2
        df = self.data.transpose(dim, ...).to_pandas()
        # noinspection PyTypeChecker
        df = df.rolling(window, win_type=win_type, center=center, min_periods=min_periods).mean()
        return Stack.from_dataframe(df).transpose(*self.dims)

    @_expose
    def median(self, *args, **kwargs):
        pass

    # noinspection PyArgumentList, PyTypeChecker
    def log10(self, clip_min=None):
        s = self
        if clip_min is not None:
            # noinspection PyNoneFunctionAssignment
            s = self.clip(min=clip_min)

        return s.__class__.from_array(
            np.log10(s.values),
            coords=s.coords,
        )

    def abs(self):
        return Stack(np.abs(self.data))

    @_expose
    def min(self, *args, **kwargs):
        pass

    @_expose
    def max(self, *args, **kwargs):
        pass

    @_expose
    def argmin(self, *args, **kwargs):
        pass

    @_expose
    def argmax(self, *args, **kwargs):
        pass

    @_expose
    def quantile(self, *args, **kwargs):
        pass

    def idxmax(self, dim):
        """get the coordinates for the maximum value along a single dimension"""
        idcs = self.data.argmax(dim)

        return Stack.from_array(
            self.data.coords[dim].values[idcs.values],
            idcs.coords,
        )

    def idxmin(self, dim):
        """get the coordinates for the minimum value along a single dimension"""
        idcs = self.data.argmin(dim)

        return Stack.from_array(
            self.data.coords[dim].values[idcs.values],
            idcs.coords,
        )

    @_expose
    def reindex(self, *args, **kwargs):
        pass

    @_expose
    def sortby(self, *args, **kwargs):
        pass

    @_expose
    def where(self, *args, **kwargs):
        pass

    @_expose
    def __eq__(self, *args, **kwargs):
        pass

    @_expose
    def __ne__(self, *args, **kwargs):
        pass

    @_expose
    def __ge__(self, *args, **kwargs):
        pass

    @_expose
    def __gt__(self, *args, **kwargs):
        pass

    @_expose
    def __le__(self, *args, **kwargs):
        pass

    @_expose
    def __lt__(self, *args, **kwargs):
        pass

    @_expose
    def __add__(self, *args, **kwargs):
        pass

    @_expose
    def __radd__(self, *args, **kwargs):
        pass

    @_expose
    def __sub__(self, *args, **kwargs):
        pass

    @_expose
    def __rsub__(self, *args, **kwargs):
        pass

    @_expose
    def __mul__(self, *args, **kwargs):
        pass

    @_expose
    def __rmul__(self, *args, **kwargs):
        pass

    @_expose
    def __truediv__(self, *args, **kwargs):
        pass

    @_expose
    def __rtruediv__(self, *args, **kwargs):
        pass

    @_expose
    def __floordiv__(self, *args, **kwargs):
        pass

    @_expose
    def __rfloordiv__(self, *args, **kwargs):
        pass

    @_expose
    def __neg__(self, *args, **kwargs):
        pass

    @_expose
    def __len__(self, *args, **kwargs):
        """returns the length along the first dimension"""
        pass

    @_expose
    def __array__(self, *args, **kwargs):
        """compatibility with numpy.asarray"""
        pass

    ######################################################################
    # Convenient data access via DataArray core api

    def sel(self, drop=True, *args, **kwargs):
        """
        Select elements by their coordinates.
        See xarray doc.
        Only difference is that scalar coordinates are dropped by default.
        """
        return Stack(self.data.sel(*args, **kwargs, drop=drop))

    def isel(self, drop=True, *args, **kwargs):
        """
        Select elements by their coordinates.
        See xarray doc.
        Only difference is that scalar coordinates are dropped by default.
        """
        return Stack(self.data.isel(*args, **kwargs, drop=drop))

    @functools.wraps(xr.DataArray.transpose)
    def transpose(self, *dims):
        """reorder dimensions"""
        transposed = self.data.transpose(*dims, transpose_coords=True)

        # make sure we keep the same order in the dictionary
        return Stack.from_array(
            transposed.values,
            {d: self.data.coords.indexes[d] for d in transposed.dims}
        )

    @_expose
    def get_axis_num(self, *args, **kwargs):
        pass

    @property
    def values(self):
        return self.data.values

    @property
    def coords(self):
        # We are not strictly following the xarray structure here.
        # We are generally not interested in non-dimensional coordinates.
        # Also, this avoids single scalar DataArrays when doing something like:
        # stack.coords['time'].min()
        return self.data.coords.indexes

    @property
    def shape(self):
        return self.data.shape

    @property
    def dims(self):
        return self.data.dims

    @property
    def ndim(self):
        return self.data.ndim

    def get_coords_except(self, *dims) -> xarray.core.coordinates.DataArrayCoordinates:
        """
        Get a DataArrayCoordinates object with the given dimensions dropped.
        This is useful when building a new Stack from a reduction of a previous one.

        :param dims: list of str names
        :return:
        """
        return self.data.reset_index(dims, drop=True).coords

    def get_coords_names(self):
        """
        Get the name of all dimensions
        """
        return list(self.coords.keys())

    def get_coords_names_except(self, *dims):
        """
        Get the name of all dimensions except the given ones.
        This is useful when we know one dimension is time but the other might change.

        :param dims: list of str names
        :return:
        """
        return list(self.get_coords_except(*dims).keys())

    def replace_dim(self, old_name, new_name, values):
        """change the name and coordinates of a particular dimension"""
        data = self.data.rename({old_name: new_name})

        if isinstance(values, pd.Index):
            # xarray looks at names that are carried in indices
            # drop any pandas structure
            values = values.values

        data.coords[new_name] = values
        return Stack(data)

    def rename_dim(self, old_name, new_name):
        """change the name of a particular dimension"""
        return self.replace_dim(
            old_name,
            new_name,
            self.coords[old_name]
        )

    def reset_coord(self, dim='time', q=0):
        """
        Reset one of the coordinates to it starts at 0.
        If you are reseting time after taking a slice you probably want shift

        :param dim: name of the dimension to reset
        :param q:
            Quantile of the current coordinates to use as a reference.
            0 makes this stack start at t=0
            1 makes this stack have negative timestamps, reaching 0 at the end
            0.5 centers this stack.
        """

        new_vals = self.coords[dim] - np.quantile(self.coords[dim], q)

        return self.replace_dim(
            dim,
            dim,
            new_vals
        )

    def sel_between(self, *, reset=False, **kwargs):
        """
        short-hand to select ranges along multiple dimensions

        example:
            select_between(mean, time=(-5, +10), channel=(0, 200))

        :param kwargs:

        :param reset:
            Whether the respective coordinates should be shifted
            to the start of each window.

            For example when to selec a time window, and have the
            resulting stack starting at t=0
        :return:
        """
        sel = {}
        for dim, (start, stop) in kwargs.items():
            sel[dim] = (start <= self.coords[dim]) & (self.coords[dim] <= stop)

        stack = self.isel(**sel)

        if isinstance(reset, bool):
            reset = None if not reset else 0.

        if reset is not None:
            # assert isinstance(reset, (float, int))

            shifts = {
                dim: -1 * ((stop - start) * float(reset) + start)
                for dim, (start, stop) in kwargs.items()
            }

            stack = stack.shift_coord(**shifts)

        return stack

    def shift_coord(self, **kwargs):
        """
        Shift coordinates of the given dimentions by a fixed offset
        Useful to reset time right after taking a slice, so we begin with t=0

            selection = stack.sel(time=win).shift_coord(time=-time.start)

        :param kwargs:
        :return:
        """
        stack = self

        for k, shift in kwargs.items():
            stack = stack.replace_dim(k, k, stack.coords[k] + shift)

        return stack

    def sel_centered(self, duration, dim='time', reset=False):
        """Cut a section in the temporal center of this stack"""
        return self.sel_between(
            **{
                dim: self.get_rel_win().take_centered(duration)
            },
            reset=reset,
        )

    def sel_pairs(self, dim_index, dim_values, pairs: pd.Series):
        """
        Select traces for pairs <cell, best-channel>.
        This avoids vectorized indexing from xarray.
        Example:

            cell_spike_shapes.sel_pairs('cell', 'channel', cells['best_chan'])

        will return a stack with "cell" in coords, where the values are the original values,
        for each cell taken at its corresponding "best_channel".

        :param dim_index: dim name that is matched to the pairs index
        :param dim_values:
        :param pairs:

        :return: a stack where "dim_values" has been removed,
        and one trace is taken for each value in pairs.
        """

        if isinstance(pairs, dict):
            pairs = pd.Series(pairs)

        # noinspection PyUnresolvedReferences
        res = self.data.sel({
            dim_index: xr.DataArray(pairs.index.values, dims=[dim_index]),
            dim_values: xr.DataArray(pairs.values, dims=[dim_index]),
        })

        res = res.drop_vars(dim_values)

        # res = res.reset_index(dim_values, drop=True)

        return Stack(res)

    def store_hdf(self, path: str, key, overwrite=False):
        """
        save a stack in an HDF5 file
        """
        with h5py.File(path, mode='a') as f:

            for dim_idx, (dim_name, dim_values) in enumerate(self.coords.items()):

                idx_key = f'{key}_index_{dim_idx:06d}'
                if idx_key in f.keys() and overwrite:
                    del f[idx_key]

                dset = f.create_dataset(idx_key, data=dim_values)
                dset.attrs['name'] = dim_name

            values_key = f'{key}_values'
            if values_key in f.keys() and overwrite:
                del f[values_key]

            _ = f.create_dataset(values_key, data=self.values)

    ######################################################################
    # Custom load/save/extract (compatible with existing stored data)

    @classmethod
    def load_hdf(cls, path: str, key, sel=None, adjust_gain=None):
        """
        Load a stack stored in HDF5
        :param path:
        :param key:
        :param sel: optional. Dictionary of <dim, coords> to specify a slice of the tensor to load.
        Note that this will be in *coords*, not in indices.
        Slices are accepted.

        :param adjust_gain:
            We may store the data in the gain un-adjusted int16 format, which takes less disk space.
            Pass here the conversion factor to produce a float stack in uV.

                load_hdf(..., adjust_gain=raw.meta['conversion_factor'])

        Example:
            # for a stored stack of rank 3: (374 channel x 13,840 peak x 1,500 time)
            load_stack_hdf(stored_data_path, 'waveshapes', ix=dict(channel=50, peak=slice(20, 30)))
            # will return a stack of rank 2: (10 peak x 1,500 time)

        :return:
        """

        sel = {} if sel is None else sel

        # reconstruct sel with the correct all dimensions in the correct order
        full_sel = []

        with h5py.File(path, mode='r') as f:
            values_dset = f[f'{key}_values']

            coords = {}
            for dim_idx in range(len(values_dset.shape)):
                index_dset = f[f'{key}_index_{dim_idx:06d}']
                name = index_dset.attrs['name']
                index = np.array(index_dset)

                if name not in sel:
                    coords_sel = slice(None)
                else:
                    coords_sel = sel[name]
                    if isinstance(coords_sel, slice):
                        coords_sel = coords_sel.indices(len(index))

                    coords_sel = np.isin(index, coords_sel)

                coords[name] = index[coords_sel]
                full_sel.append(coords_sel)

            values_dset = f[f'{key}_values'][tuple(full_sel)]
            values = values_dset

        stack = cls.from_array(values, coords)

        if adjust_gain is not None:
            if not np.issubdtype(stack.values.dtype, np.integer):
                logging.warning('gain already adjusted?')

            channels = stack.coords['channel'].values
            gain_adjust = Stack.from_array(adjust_gain[channels], {'channel': channels})
            stack = stack * gain_adjust

        return stack

    @classmethod
    def load_idcs(
            cls, raw, windows,
            channels='all', load_hz=None, adjust_gain=True, show_pbar=True,
            early_exit=False,
    ):
        """
        Load one section per channel and window.

        :param raw:
        :param windows:
            All of the windows specified in sample indices (integer).
            All windows must be of exactly the same length in number of samples.

        :param channels: single channel or list of channel idcs. Default: all valid channels.
        :param load_hz:
        :param adjust_gain:
        :param show_pbar:
        :param early_exit:
        :return: a Stack of shape (channel, window, sample)
        """
        if isinstance(windows, timeslice.Windows):
            assert windows.are_in_samples()
            windows = windows.wins
        assert isinstance(windows, pd.DataFrame)

        length = windows['stop'] - windows['start']
        assert len(length.drop_duplicates()) == 1
        length = length.iloc[0]

        if isinstance(channels, str) and channels == 'all':
            channels = raw.channels.index

        stored_hz = raw.sampling_rate

        if load_hz is None:
            load_hz = stored_hz

        load_hz = timeslice.match_load_hz(stored_hz, load_hz, thresh=.1)
        timeslice.assert_stride(stored_hz, load_hz, 'stored_hz', 'load_hz')
        load_stride = timeslice.get_stride(stored_hz, load_hz)

        length = int(np.ceil(length / load_stride))

        if adjust_gain:
            dtype = np.float64
        else:
            dtype = np.int16

        traces = np.zeros((len(channels), len(windows), length), dtype=dtype)

        combs = list(itertools.product(
            enumerate(channels),
            enumerate(windows[['start', 'stop']].itertuples())
        ))

        if show_pbar:
            combs = tqdm(combs, desc='load trace')

        if (windows['start'] < 0).any() or (windows['stop'] < 0).any():
            logging.warning(f'Negative indices. Too close to edge?')

        try:
            for (i, chan), (j, (win_idx, start, stop)) in combs:
                sec = raw.load(
                    sample_idcs=slice(start, stop, load_stride),
                    channels=[chan],
                    adjust_gain=adjust_gain,
                )

                traces[i, j] = sec

        except KeyboardInterrupt:
            if early_exit:
                print('early exit', flush=True)
            else:
                raise

        sample_idcs = np.arange(
            windows['start'].iloc[0] - windows['ref'].iloc[0],
            windows['stop'].iloc[0] - windows['ref'].iloc[0],
            load_stride,
        )

        win_dim = 'window' if windows.index.name is None else windows.index.name

        coords = {
            'channel': channels,
            win_dim: windows.index,
            'sample': sample_idcs,
        }

        return cls.from_array(traces, coords)

    @classmethod
    def load_ms(
            cls, raw, times, win_ms,
            channels='all', load_hz=None, adjust_gain=True, show_pbar=True,
            early_exit=False,
    ):
        """
        Extract one section per channel and window.

        :param raw:
        :param channels:
        :param times: The reference times. A section will be extracted for each.
        :param win_ms: A tuple (pre, post) to be applied around the given times.
            Note that this time defintion won't be fully respected because it is specified
            as a floating value, but the raw data is discrete (according to a sampling rate).
            Depending on how the combination of times, load_hz and win_ms aligns
            with the original sampling rate of the data, we may need to round up or down
            each window.
            Because all the resulting stack must have exactly the same length in number of samples
            for each section, these windows will be cropped to match.

        :param load_hz:
        :param adjust_gain:
        :param show_pbar:
        :param early_exit:

        :return: a Stack of shape (channel, window, ms)
        """
        wins_ms = timeslice.Windows.build_around(times, win_ms)
        wins_idcs = wins_ms.ms_to_sample(raw.sampling_rate)
        wins_idcs = wins_idcs.crop_to_minimum_common()

        stack = cls.load_idcs(
            raw, windows=wins_idcs, channels=channels,
            load_hz=load_hz, adjust_gain=adjust_gain, show_pbar=show_pbar,
            early_exit=early_exit,
        )

        return stack.reindex_sample_to_ms(raw.sampling_rate)

    @classmethod
    def load_single_ms(
            cls, raw, load_win=None, stored_hz=None, load_hz=None,
            channels='all', adjust_gain=True
    ):
        """
        Extract a single section per channel.

        :param raw:
        :param load_win: a pair of start, stop given in ms
        :param stored_hz:
        :param load_hz:
        :param channels: single channel or list of channel idcs. Default: all valid channels.
        :param adjust_gain:

        :return: a Stack of shape (channel, ms)
        """
        if stored_hz is None:
            stored_hz = raw.sampling_rate

        if load_win is None:
            load_win = raw.win_ms

        assert load_win[0] >= 0, f'Expected absolute time. Got: {load_win}'
        assert load_win[1] >= 0, f'Expected absolute time. Got: {load_win}'

        load_win = timeslice.Win(
            timeslice.ms_to_idcs(stored_hz, load_win[0]),
            timeslice.ms_to_idcs(stored_hz, load_win[1])
        )

        stack = cls.load_single_idcs(
            raw, load_win,
            stored_hz=stored_hz,
            load_hz=load_hz,
            channels=channels,
            adjust_gain=adjust_gain,
        )

        return stack.reindex_sample_to_ms(raw.sampling_rate)

    @classmethod
    def load_single_idcs(
            cls, raw, load_win=None, stored_hz=None,
            load_hz=None, channels='all',
            adjust_gain=True
    ):
        """
        Extract a single section per channel.

        :param raw:
        :param load_win: a pair of start, stop given in sample indices
        :param stored_hz:
        :param load_hz:
        :param channels: single channel or list of channel idcs. Default: all valid channels.
        :param adjust_gain:

        :return: a Stack of shape (channel, ms)
        """
        if load_win is None:
            load_win = raw.win_idcs

        single_channel = np.issubdtype(type(channels), np.number)

        if isinstance(channels, str) and channels == 'all':
            channels = raw.channels.index

        if stored_hz is None:
            stored_hz = raw.sampling_rate

        if load_hz is None:
            load_hz = stored_hz

        load_hz = timeslice.match_load_hz(stored_hz, load_hz, thresh=.1)
        timeslice.assert_stride(stored_hz, load_hz, 'stored_hz', 'load_hz')
        stride_i = timeslice.get_stride(stored_hz, load_hz)

        load_win = timeslice.Win(*load_win)
        load_win = load_win.clip(raw.win_idcs)
        load_slice = slice(load_win.start, load_win.stop, stride_i)

        main = raw.load(
            channels=channels,
            sample_idcs=load_slice,
            adjust_gain=adjust_gain,
        )

        idcs = np.arange(load_slice.start, load_slice.stop, load_slice.step)

        coords = {}

        # drop channel dimension if we were asked for a particular channel (scalar)
        if single_channel:
            main = main[0]
        else:
            coords['channel'] = channels

        coords['sample'] = idcs

        return cls.from_array(main, coords)

    ######################################################################
    # Other convenience methods

    def reindex_sample_to_ms(self, sampling_rate, dim='sample', new_dim='time'):
        """
        If one of the dimensions of this stack is specified in "sample indices",
        convert it to time in miliseconds according to a given sampling rate
        """
        assert dim in self.dims

        idcs = self.coords[dim].values
        # noinspection PyTypeChecker
        assert np.issubdtype(idcs.dtype, np.integer)

        period = S_TO_MS / sampling_rate

        # try to stay integer for as long as possible to avoid floating errors
        # caussing problems when indexing by time
        if period.is_integer():
            period = int(period)

        time = idcs * period

        return self.replace_dim(dim, new_dim, time)

    def reindex_ms_to_sample(self, sampling_rate, dim='time', new_dim='sample'):
        """
        If one of the dimensions of this stack is specified in "sample indices",
        convert it to time in miliseconds according to a given sampling rate
        """
        assert dim in self.dims

        this_period = self.estimate_sampling_period(dim)

        idcs = self.coords[dim].values
        period = S_TO_MS / sampling_rate
        if period.is_integer():
            period = int(period)

        if (this_period % period) > 1e-6:
            logging.error(
                f'Sampling period missalignment of {this_period % period} '
                f'(stack: {this_period}; data: {period})')

        samples = np.round(idcs / period).astype(int)

        return self.replace_dim(dim, new_dim, samples)

    def estimate_sampling_period(self, dim='time') -> float:
        """
        Look for a dimension called 'time',
        assume it is in ms and uniformly sampled,
        return sampling period (ms)
        """
        tstep = np.diff(self.coords[dim].values)
        assert np.allclose(tstep, tstep[0]), \
            f'Expected regular {dim} steps. ' \
            f'Got: {np.unique(np.round(tstep, decimals=9))}'

        tstep = tstep[0]

        # 'allclose' can let pass small rounding errors
        # remember time unit is milliseconds, so we are going
        # to round up to 1 pico second
        tstep = timeslice.adjust_sampling_period(tstep, quiet=True)

        if float(tstep).is_integer():
            tstep = int(tstep)

        return tstep

    def estimate_sampling_rate(self, dim='time') -> float:
        """
        Look for a dimension called 'time',
        assume it is in ms and uniformly sampled,
        return sampling freq (Hz)
        """
        tstep = self.estimate_sampling_period(dim=dim)
        sampling_rate = 1. / (tstep * MS_TO_S)

        if sampling_rate.is_integer():
            sampling_rate = int(sampling_rate)

        return sampling_rate

    def extend(self, dim, pre=0, post=0, fill_value=np.nan):
        """
        Extend this stack along a given dimension.
        For example before shifting, to increase the maximum possible width of the window.

        This only works for dimensions with regular coordinates (e.g. time
        under a constant sampling rate or depth of a probe).
        """
        step = np.diff(self.coords[dim])
        assert np.allclose(step, step[0])
        step = step[0]

        old_index = self.coords[dim].values

        pre = np.min(old_index) - (np.arange(pre)[::-1] + 1) * step
        post = np.max(old_index) + (np.arange(post) + 1) * step

        new_index = np.concatenate([pre, old_index, post])

        return self.reindex(**{dim: new_index}, fill_value=fill_value)

    def slide_template(self, ref, align=1):
        """
        Slide a template along this array and evaluate
        pearson's correlation at every step.

        :param ref:
        :param align:
        :return:
        """
        assert self.ndim == 1
        assert ref.ndim == 1

        assert self.estimate_sampling_period() == ref.estimate_sampling_period(), \
            f'Sampling periods should match for template detection.'

        dim = self.get_coords_names()[0]

        slice_length = len(ref)

        coord = self.coords[dim]
        n = len(coord)

        idcs = np.arange(0, n - slice_length)

        scores = np.empty(len(idcs))

        for i in tqdm(idcs, desc='slide'):
            candidate = self.values[i:i + slice_length]

            r, p_value = scipy.stats.pearsonr(candidate, ref.values)

            scores[i] = r

        idx_offset = int(np.round((slice_length - 1) * align))
        assert 0 <= idx_offset < slice_length

        return self.__class__.from_array(scores, {dim: coord[idcs + idx_offset]})

    def crop(self, times, win_ms, pbar=None, new_dim='sample'):
        """
        Given several time points in this 1-D stack and a window, extract a new 2-D stack from it.
        This is equivalent to "load" a stack from another.

        We assume: the stack is 1D, all the times values are *exactly* found in this stack's time coord.

        :param times:
        :param win_ms:
        :param new_dim:
        :param pbar:

        :return: a new 2-D stack
        """

        assert self.ndim == 1

        coord_name = list(self.coords.keys())[0]
        coord = list(self.coords.values())[0]

        refs = np.searchsorted(coord, times)

        hz = self.estimate_sampling_rate()

        win_idcs = timeslice.Win(
            timeslice.ms_to_idcs(hz, win_ms[0]),
            timeslice.ms_to_idcs(hz, win_ms[1])
        )

        starts = refs + win_idcs.start
        stops = refs + win_idcs.stop

        valid = (starts >= 0) & (stops <= len(coord))

        starts = starts[valid]
        stops = stops[valid]

        res = np.empty((len(starts), win_idcs.length), dtype=self.dtype)

        if pbar is not None:
            starts = pbar(starts, desc='crop')

        for i, (start, stop) in enumerate(zip(starts, stops)):
            res[i] = self.values[start:stop]

        # idcs = starts[:, np.newaxis] + np.arange(win_idcs.length)
        # res = self.values[idcs]

        new_coords = {
            new_dim: times[valid],
            coord_name: timeslice.idcs_to_ms(hz, win_idcs.arange(1)),
        }

        return self.__class__.from_array(res, new_coords)

    def extract_pearsonr(self, ref, pbar=None):
        """
        Extract Pearson's correlation coefficient along the last dimension
        (typically time).

        Example to understand how similar are all traces to their average:

            all_sw.extract_pearsonr(all_sw.reduce(np.mean, ['peak', 'channel']))

        :param ref: a 1D stack or a numpy array
        :param pbar:

        :return: a new stack where the last dimension is dropped (rank = self.rank - 1)
        """
        iter_dim = self.dims[0]
        dims = self.dims[1:]
        if dims != ('time',):
            logging.warning(f'Extracting pearson r along last dimension(s): {dims}')

        if isinstance(ref, (xr.DataArray, Stack)):
            ref = ref.values

        assert self.values.shape[1:] == ref.shape

        items = self.values.reshape(self.values.shape[0], -1)
        ref = ref.ravel()

        assert items.shape[1:] == ref.shape

        if pbar is not None:
            items = pbar(items, desc='pearsonr')

        corr = np.array([
            scipy.stats.pearsonr(a, ref)[0]
            for a in items
        ])

        # noinspection PyTypeChecker
        return Stack.from_array(
            corr,
            {iter_dim: self.coords[iter_dim]},
        )

    def cross_corr(self, ref, dim='time'):
        """
        build a cross correlation stack relative to the given reference

        Useful to pick up delays between signals. To get the relative delays in ms:

            stack.cross_corr(stack.mean('gids')).idxmax('time')

        :param ref:
        :param dim:
        :return:
        """

        assert self.ndim == 2

        if isinstance(ref, Stack):
            ref = ref.data

        if isinstance(ref, xr.DataArray):
            assert ref.ndim == 1
            assert dim in ref.dims, f'Expected {dim} got {ref.dims} on ref dims'
            ref = ref.values

        ms_step = timeslice.S_TO_MS / self.estimate_sampling_rate(dim)
        t = (np.arange(len(self.coords[dim])) - len(self.coords[dim]) // 2) * ms_step

        s = self.transpose(..., dim)
        other = s.dims[0]

        corr = []

        for i in range(s.shape[0]):
            corr.append(np.correlate(s.values[i], ref, mode='same'))

        corr = Stack.from_array(np.array(corr), {other: s.coords[other], dim: t})

        return corr.transpose(*self.dims)

    def extract_spectrogram_1d(self, segment_ms=1_000, overlap_ms=None):
        """
        Extract the spectrogram of this 1D trace
        """
        assert self.ndim == 1

        period = self.estimate_sampling_period()
        nperseg = int(segment_ms / period)
        nperseg = int(np.clip(nperseg, 1, np.inf))

        if overlap_ms is None:
            overlap_ms = segment_ms * .95
        noverlap = int(overlap_ms / period)
        noverlap = int(np.clip(noverlap, 0, nperseg - 1))

        freqs, time, spec = scipy.signal.spectrogram(
            self.values,
            fs=self.estimate_sampling_rate(),
            nperseg=nperseg,
            noverlap=noverlap,
        )

        time_offset = self.coords['time'].min()
        time = time * S_TO_MS + time_offset

        spec = Stack.from_array(spec, coords={'freq': freqs, 'time': time})
        spec_db = 10 * spec.log10()

        return spec_db

    def extract_spectrogram(self, segment_ms=1_000, overlap_ms=None, dim='time'):
        """
        Extract the spectrogram along the time dimention.
        All other dimensions are treated as independent traces.
        """

        reg = []
        data = []

        if len(self.shape) == 1:
            return self._extract_spectrogram_1d(segment_ms=segment_ms, overlap_ms=overlap_ms)

        for sel, subs in self.iter_except(dim, pbar=tqdm):
            res = subs.extract_spectrogram_1d(segment_ms=segment_ms, overlap_ms=overlap_ms)

            reg.append(sel)
            data.append(res)

        return stackup_multidim(data, pd.DataFrame(reg))

    def histogram_along(self, dim: str, new: str, bins=100, quiet=False):
        """
        Extract a histogram for every single value along a dimension.

        :param dim:
        :param new: name
        :param bins:
        :param quiet:

        :return: a 2D stack of shape len(coords[dim]) x len(bins)
        """
        if isinstance(bins, int):
            bins = np.linspace(np.min(self.values), np.max(self.values), bins + 1)

        values = self.coords[dim].values
        if not quiet:
            values = tqdm(values, desc=str(dim))

        hs = np.array([
            np.histogram(self.sel(**{dim: v}).values.ravel(), bins=bins)[0]
            for v in values
        ])

        bin_centers = .5 * (bins[1:] + bins[:-1])

        hs = self.__class__.from_array(
            hs,
            coords={
                dim: self.coords[dim],
                new: bin_centers
            })

        return hs

    def reset_baseline(self, between_ms=(-np.inf, +np.inf), dim='time'):
        """
        subtract the median value, of each independent trace,
        obtained from a range of time
        """
        # noinspection PyTypeChecker
        # noinspection PyNoneFunctionAssignment
        baseline = self.sel_between(**{dim: between_ms}).median(dim)
        return self - baseline

    def sample(self, **sel):
        """
        randomly pick a fixed count of traces along a dimension without replacement
        eg:
            stack.sample(channel=4, spike=10)  # gives you 10 spikes over 4 channels, randomly picked
        """
        masks = {}

        for dim, coord in self.coords.items():
            dim: str

            mask = np.ones(len(self.coords[dim]), dtype=np.bool_)

            if dim in sel.keys():
                mask[sel[dim]:] = False
                np.random.shuffle(mask)

            masks[dim] = mask

        return self.isel(**masks)

    def split_by(self, dim, values: np.ndarray, replace: np.ndarray = None) -> dict:
        """

        Split this stack in multiple that have the same value along a given dimention.
        Similar to groupby but result is a dictionary.

        Useful to split a stack of many channels per probe:

            ch_details = true_raw.get_channel_details()
            data_per_probe = data.split_by('channel', ch_details['probe'], ch_details['ch'])


        :param dim:
        :param values:
        :param replace: Replace values in dim for these ones.
        :return:
        """
        if isinstance(values, pd.Series):
            values = values.reindex(self.coords[dim]).values

        if isinstance(replace, pd.Series):
            replace = replace.reindex(self.coords[dim]).values

        assert not np.any(np.isnan(values))

        split = {}

        for v in np.unique(values):
            mask = values == v
            split[v] = self.isel(**{dim: mask})

            if replace is not None:
                split[v] = split[v].replace_dim(dim, dim, replace[mask])

        return split

    def groupby_array(self, dim, new_name, values: np.ndarray):
        """
        short-hand to groupby an external array of values that has
        the same shape as one of our dimensions.

        This will select multiple groups of traces along a single dimension and reduce
        each group to a single trace.  example, if a stack represents spike wave forms (spike_index x time)
        and we have metadata on the spikes (their time in the experiment)
        we can group traces based on that metadata (group spikes in chunks of 5 minutes, so we see
        if there is an evolution in the recording over time)

        call like:
            meta = pd.cut(spikes['time'], bins=np.arange(0, spikes['time'].max() + bin_width, bin_width))

            restacked = Stack(all_sw.groupby_array('peak', 'period', meta).mean())

        Note that the returned object will be an xr.DataArrayGroupBy, so the result of mean() will
        be an xr.DataArray, which means we need to cast it back.

        :param dim: dimension to be matched to "values"
        :param new_name: new name of the dimension after reduction
        :param values: array of values that indicates which traces are grouped together.
            if a pd.Series, it will be reindexed by the 'dim' coordinate.
        :return:
        """
        if isinstance(values, pd.Series):
            values = values.reindex(self.coords[dim]).values

        return self.data.groupby(
            xr.DataArray(
                np.asarray(values),
                dims=[dim],
                coords={dim: self.data.coords[dim]},
                name=new_name,
            ))

    def groupby_mean(self, dim, new_name, values: np.ndarray):
        """
        A short-hand to groupby values and take the mean over the dimension, which is the common case.
        We do this often, for example, to obtain the average spike shape for a cell:

            all_spike_shapes.groupby_mean('spike_id', 'cell', spks.spikes['gid'])

        :return: a Stack object
        """
        mean_shapes = self.groupby_array(dim, new_name, values).mean()
        return Stack(mean_shapes)

    def groupby_std(self, dim, new_name, values: np.ndarray):
        """
        A short-hand to groupby values and take the std over the dimension, which is the common case.
        We do this often, for example, to obtain the average spike shape for a cell:

            all_spike_shapes.groupby_std('spike_id', 'cell', spks.spikes['gid'])

        :return: a Stack object
        """
        std_shapes = self.groupby_array(dim, new_name, values).std()
        return Stack(std_shapes)

    def apply_shift(self, shift_steps, by='spike_id', on='time'):
        """
        Extract a stack from within this one where samples are shifted by a different ammount
        for each value of the "by" direction on the "on" dimension.
        For example, if you have a stack of traces representing spike shapes over multiple channels:
            (spike_id, channel, time)
        and a series of temporal shifts (specified in # samples, not in ms!), you can apply it to this stack:
        apply_shift(shifts, by='spike_id', on='time')

        Note that the resulting stack will be smaller, limited by the most strict shifts (up or down).
        The new "time" coordinate will have its values taken from this strict window, so that
        any items with shift 0 are left intact (just cropped).

        Shift sign indicates the direction that each row moves:

            positive
                = move to the left
                = the event happens earlier
                = we crop late data
                = as if we had extracted at t + |shift|

            negative
                = as if we had extracted at t - |shift|

            We're taking this convetion to be compatible with the affinewrap library.

        Example:

            fake_shifts = np.zeros(3)
            fake_shifts[0] = -2

            fake_stack = stacks.Stack.from_array(
                np.eye(3, 5) + np.eye(3, 5)[:, ::-1],
                {'spike_id': [0, 1, 2], 'time': [-2, -1, 0, 1, 2]}
            )

            # original stack:
            # array([     [1., 0., 0., 0., 1.],
            #             [0., 1., 0., 1., 0.],
            #             [0., 0., 2., 0., 0.]])
            #
            # time: array([-2, -1,  0,  1,  2])

            shifted = fake_stack.apply_shift(fake_shifts)

            # shifted stack:
            # array([     [1., 0., 0.],
            #             [0., 1., 0.],
            #             [2., 0., 0.]])
            #
            # time: array([ 0,  1,  2])

        :param shift_steps:
        :param by:
        :param on:
        :return:
        """

        if np.issubdtype(type(shift_steps), np.number):
            shift_steps = np.ones(len(self.coords[by])) * shift_steps

        if isinstance(shift_steps, pd.Series):
            shift_steps = shift_steps.reindex(self.coords[by]).values

        shift_steps = np.asarray(shift_steps)
        assert len(shift_steps) == len(self.coords[by])

        if np.any(np.isnan(shift_steps)):
            logging.warning(f'{np.sum(np.isnan(shift_steps))} nans in shift values. Not aligned to coords?')
            shift_steps[np.isnan(shift_steps)] = 0

        # noinspection PyTypeChecker
        crop_range_rel = (
            -int(min(np.nanmin(shift_steps), 0)),
            -int(max(np.nanmax(shift_steps), 0)),
        )

        old_len = len(self.coords[on])

        crop_range = crop_range_rel[0], crop_range_rel[1] + old_len

        rearranged = self.transpose(..., by, on)

        main_idcs = np.arange(*crop_range)
        main_idcs = main_idcs + shift_steps[:, np.newaxis]
        main_idcs = main_idcs.astype(np.integer)

        matching_shape = (-1,) * (rearranged.values.ndim - 2) + main_idcs.shape

        main_idcs = main_idcs.reshape(matching_shape)
        assert main_idcs.ndim == rearranged.ndim

        shifted = np.take_along_axis(rearranged.values, main_idcs, axis=-1)

        coords = dict(rearranged.coords)
        coords[on] = coords[on][crop_range[0]:crop_range[1]]

        return Stack.from_array(shifted, coords).transpose(*self.dims)

    def zscore(self, dim='time'):
        """
        return a version of this stack z-scored: (self - self.mean('time')) / self.std('time')

        :param dim:
        :return:
        """
        # unfortunately not valid in current scipy version:
        # zscored = scipy.stats.zscore(self.values, axis=self.get_axis_num(dim), nan_policy='omit')
        # so need to deploy our own to handle nans:

        # noinspection PyTypeChecker
        # noinspection PyNoneFunctionAssignment
        axis = self.get_axis_num(dim)

        mean = np.nanmean(self.values, axis=axis)
        std = np.nanstd(self.values, axis=axis)

        broadcast_index = [slice(None)] * len(self.values.shape)
        # noinspection PyTypeChecker
        broadcast_index[axis] = np.newaxis
        broadcast_index = tuple(broadcast_index)

        zscored = (self.values - mean[broadcast_index]) / std[broadcast_index]

        # noinspection PyTypeChecker
        return Stack.from_array(
            zscored,
            self.coords,
        )

    def zscore_rolling(self, win_count, dim='time', dropna=True):
        """
        return a version of this stack z-scored: (self - self.mean('time')) / self.std('time')
        but using a sliding window of size "window"

        :param win_count: an INTEGER indicating the number of samples to include in the window.
        For example, you will probably want to do something like: int(self.estimate_sampling_rate() * 60 * 4)
        to use a sliding window of ~4 min. Notice that this depends on the sampling rate of the current stack

        :param dropna:
        :param dim:
        :return:
        """
        assert np.issubdtype(type(win_count), np.integer)

        def pd_roll(x, **kwargs):
            r = x.rolling(**kwargs)
            m = r.mean().shift(1)
            s = r.std(ddof=0).shift(1)
            z = (x - m) / s
            return z

        rolled = Stack(
            xr.DataArray.from_series(
                pd_roll(
                    self.data.to_series().unstack(dim).T,
                    window=win_count,
                    center=True,
                    min_periods=1
                ).T.stack()
            )
        )

        # ensure same order as before
        for k, vs in self.coords.items():
            # noinspection PyNoneFunctionAssignment
            rolled = rolled.reindex(**{k: vs})
        rolled.transpose(*self.dims)

        if dropna:
            # noinspection PyTypeChecker
            # noinspection PyNoneFunctionAssignment
            rolled = rolled.dropna(dim)

        return rolled

    def get_rel_win(self, dim='time'):
        """
        Get the period (start, stop) covered by this stack in a 'time' dimension
        """
        return timeslice.Win(
            np.min(self.coords[dim].values),
            np.max(self.coords[dim].values),
        )

    def downsample(self, downsample_hz, dim='time'):
        """
        Try to match the new sampling frequency by taking every nth sample (including the first)
        Note the frequency should be as close as possible to a perfect divisor of the current sampling frequency.
        See timeslice.get_stride
        """
        sampling_hz = self.estimate_sampling_rate()
        timeslice.assert_stride(sampling_hz, downsample_hz)
        factor = timeslice.get_stride(sampling_hz, downsample_hz)
        return self.downsample_by_factor(factor, dim=dim)

    def downsample_by_factor(self, factor: int, dim='time'):
        """keep every nth sample including the first"""
        idcs = np.arange(0, len(self.coords[dim]), factor)
        return self.isel(**{dim: idcs})

    def upsample(self, upsample_hz, win=None, dim='time', kind='linear'):
        """
        Upsample a stack by linearly interpolating along one dimension

        :param upsample_hz:
        :param win:
        :param dim:
        :param kind:
        :return:
        """
        if win is None:
            win = self.get_rel_win(dim)

        new_sampling_period = S_TO_MS / upsample_hz

        current_hz = self.estimate_sampling_rate(dim=dim)
        if upsample_hz < current_hz:
            logging.warning(f'Downsampling signal instead of upsampling ({upsample_hz}hz < {current_hz}hz)')

        new_idcs = np.arange(*win, new_sampling_period)

        return self.interp(new_idcs, dim=dim, kind=kind)

    def interp_between(self, win: timeslice.Win, step, **kwargs):
        win = timeslice.Win(*win)
        new_idcs = win.arange(step)
        return self.interp(new_idcs, **kwargs)

    def interp(self, new_idcs: np.ndarray, dim='time', kind='linear', pbar=None):
        """
        Interpolate a stack along one dimension

        :param new_idcs:
        :param dim:
        :param kind:
        :param pbar:
        :return:
        """
        assert (self.coords[dim].min() <= np.min(new_idcs)), \
            f'New {dim} starts at {np.min(new_idcs)} but first sample is at {self.coords[dim].min()}'

        assert (np.max(new_idcs) <= self.coords[dim].max()), \
            f'New {dim} stops at {np.max(new_idcs)} but last sample is at {self.coords[dim].max()}'

        import scipy.interpolate
        current = self.coords[dim].values

        new_vals = []

        for sel, trace in self.iter_except(dim, pbar=pbar):
            stack_lerp = scipy.interpolate.interp1d(
                current,
                trace.values,
                kind=kind,
            )

            resampled = stack_lerp(new_idcs)

            new_vals.append(resampled)

        new_coords = self.get_coords_except('time')

        new_coords = {k: v.values for k, v in new_coords.items()}

        new_coords[dim] = new_idcs

        shape = tuple(len(v) for v in new_coords.values())

        new_vals = np.concatenate(new_vals).reshape(shape)

        return Stack.from_array(new_vals, new_coords)

    def filter_pass(self, hz: tuple, dim='time'):
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

        :param dim:
        """
        if hz is None:
            return self

        low, high = hz
        low_open = low is None or np.isclose(low, 0) or np.isinf(low) or np.isnan(low)
        high_open = high is None or np.isclose(high, 0) or np.isinf(high) or np.isnan(high)

        if low_open and not high_open:
            return self.low_pass(high, dim=dim)

        elif not low_open and high_open:
            return self.high_pass(low, dim=dim)

        elif not low_open and not high_open:
            return self.band_pass(low, high, dim=dim)

        else:
            return self

    def band_pass(self, low_hz, high_hz, dim='time', *, order=2):
        sampling_hz = self.estimate_sampling_rate()

        # Set bands
        nyquist_freq = sampling_hz / 2
        low_hz = low_hz / nyquist_freq
        high_hz = high_hz / nyquist_freq

        # Calculate coefficients
        params = scipy.signal.butter(order, [low_hz, high_hz], btype='band')

        # Filter signal
        # noinspection PyTypeChecker
        filtered_data = scipy.signal.filtfilt(*params, self.values, axis=self.get_axis_num(dim))

        # noinspection PyTypeChecker
        return Stack.from_array(
            filtered_data,
            self.coords,
        )

    def low_pass(self, high_hz, dim='time', *, order=2):
        sampling_hz = self.estimate_sampling_rate()

        # Set bands
        nyquist_freq = sampling_hz / 2
        high = high_hz / nyquist_freq

        # Calculate coefficients
        params = scipy.signal.butter(order, high, btype='low')

        # Filter signal
        # noinspection PyTypeChecker
        filtered_data = scipy.signal.filtfilt(*params, self.values, axis=self.get_axis_num(dim))

        # noinspection PyTypeChecker
        return Stack.from_array(
            filtered_data,
            self.coords
        )

    def high_pass(self, low_hz, dim='time', *, order=2):
        sampling_hz = self.estimate_sampling_rate()

        # Set bands
        nyquist_freq = sampling_hz / 2
        low = low_hz / nyquist_freq

        # Calculate coefficients
        params = scipy.signal.butter(order, low, btype='high')

        # Filter signal
        # noinspection PyTypeChecker
        filtered_data = scipy.signal.filtfilt(*params, self.values, axis=self.get_axis_num(dim))

        # noinspection PyTypeChecker
        return Stack.from_array(
            filtered_data,
            self.coords,
        )

    def _find_peaks_single(self, **kwargs) -> pd.DataFrame:
        """find peaks in a 1-dimensional stack"""
        assert self.ndim == 1
        # noinspection PyTypeChecker
        dim: str = self.dims[0]

        peak_idcs, peak_props = scipy.signal.find_peaks(self.values, **kwargs)

        peaks = pd.DataFrame.from_dict(peak_props)
        peaks['sample_idx'] = peak_idcs
        peaks[dim] = self.coords[dim].values[peaks['sample_idx'].values]

        plural_cols = [
            'peak_heights', 'prominences',
            'left_bases', 'right_bases',
            'widths', 'width_heights',
            'left_ips', 'right_ips',
        ]

        peaks.rename(columns={c: c[:-1] for c in plural_cols}, inplace=True, errors='ignore')

        return peaks

    def find_peaks(self, dim='time', width_ms=None, negative=False, **kwargs) -> pd.DataFrame:
        """
        Find peaks in a signal along only one of the dimensions

        :param negative: look for minima instead of maxima

        :param dim:
        :param width_ms:
            No multiple peaks within these many ms.
            If specified, it overries 'width' of normal kwargs

        :param kwargs: parameters to scipy.signal.find_peaks

        :return: pd.DataFrame with information about detected SW. Looks like:

                  channel  peak_height  prominence  ...     right_ip  sample_idx        time
            item                                    ...
            0          23    -4.179747    5.810959  ...  5770.133325        5555  21555000.0
            1          23    -3.443084    2.170013  ...  5999.499110        5894  21894000.0
            2          23    -2.503901    5.860240  ... 18275.073024       18067  34067000.0

        """
        from functools import partial

        assert not (width_ms is not None and 'width' in kwargs)
        if width_ms is not None:
            sampling_hz = self.estimate_sampling_rate(dim)
            kwargs['width'] = width_ms * MS_TO_S * sampling_hz

        stack = self
        if negative:
            stack = -stack

        peaks = stack.apply_dataframe(
            partial(Stack._find_peaks_single, **kwargs),
            'peak',
            dim=dim,
        )

        # make sure index is unique
        peaks.reset_index(drop=True, inplace=True)
        peaks.rename_axis(columns='peak_idx', inplace=True)

        if negative and 'peak_height' in peaks.columns:
            peaks['peak_height'] *= -1

        return peaks


class StackSet:
    """
    A set of data with some metadata for each one.
    Data can be arbitrary, but most likely pd.DataFrame or Stack.
    Metadata is stored as a pd.DF table (reg) with one row per data.
    Data are stored as a dictionary with each key being the index in reg.
    Useful to select different conditions or experiments.
    """

    def __init__(self, reg: pd.DataFrame, data: dict):
        self.reg = reg
        self.data = data
        assert np.all(np.isin(reg.index, list(data.keys())))

    @classmethod
    def from_dict(cls, data: dict, names=None):
        reg = pd.DataFrame.from_records(list(data.keys()), columns=names)
        data_mapped = {uid: data[tuple(k)] for uid, *k in reg.itertuples()}
        return cls(reg, data_mapped)

    def to_hdf(self, filename):
        """
        Store data and registry to HDF5.
        Data must be pd.DataFrame or Stack
        """
        self.reg.to_hdf(filename, key='reg')
        for k, v in self.data.items():

            if hasattr(v, 'to_hdf'):
                key = f'dataframe_{k:06d}'
                v.to_hdf(filename, key=key)

            elif hasattr(v, 'store_hdf'):
                key = f'stack_{k:06d}'
                v.store_hdf(filename, key=key)

            else:
                raise NotImplementedError()

    @classmethod
    def from_hdf(cls, filename, show_pbar=True):
        """
        Load data and registry from HDF5.
        """
        # noinspection PyTypeChecker
        reg: pd.DataFrame = pd.read_hdf(filename, key='reg')

        with h5py.File(str(filename), mode='r') as f:
            file_keys = list(f.keys())

        data = {}
        index = reg.index
        if show_pbar:
            index = tqdm(reg.index, desc='loading')

        for k in index:

            fmt = 'dataframe'
            key = f'{fmt}_{k:06d}'
            if key in file_keys:
                data[k] = pd.read_hdf(filename, key=key)
                continue

            fmt = 'stack'
            key = f'{fmt}_{k:06d}'

            if f'{key}_values' in file_keys:
                data[k] = Stack.load_hdf(filename, key=key)
                continue

            raise KeyError(f'Missing data for {key}. Found: {file_keys}')

        return cls(reg, data)

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.reg._repr_html_()

    def __repr__(self):
        return self.reg.__repr__()

    def __len__(self):
        return len(self.reg)

    def simplify(self):
        """drop columns with only one value repeated"""
        simpler = self.reg[[c for c, s in self.reg.items() if s.nunique() > 1]]

        return self.__class__(
            simpler,
            self.data,
        )

    def sel(self, simplify=True, **kwargs):
        assert len(kwargs) > 0
        mask = np.array([(self.reg[key] == value) for key, value in kwargs.items()])
        mask = np.all(mask, axis=0)

        sel = self.sel_mask(mask, simplify=simplify)
        return sel

    def sel_between(self, simplify=True, **kwargs):
        mask = np.array([self.reg[key].between(*vrange).values for key, vrange in kwargs.items()])
        mask = np.all(mask, axis=0)

        sel = self.sel_mask(mask, simplify=simplify)
        return sel

    def sel_mask(self, mask, simplify=True):
        new_reg = self.reg.loc[mask]

        sel = self.__class__(
            new_reg,
            {uid: self.data[uid] for uid in new_reg.index}
        )

        if simplify:
            sel = sel.simplify()

        return sel

    def get(self, **kwargs):
        """Select a SINGLE stack and return it"""
        if len(kwargs) == 0:
            selected = self
        else:
            selected = self.sel(**kwargs)

        assert len(selected) > 0, f'No items selected'
        assert len(selected) == 1, f'Multiple items selected ({len(selected):,g})'
        return list(selected.data.values())[0]

    def items(self, show_pbar=True):
        index = self.reg.index

        pbar_desc = ''
        if isinstance(show_pbar, str):
            pbar_desc = show_pbar
            show_pbar = True

        if show_pbar:
            index = tqdm(index, total=len(self.reg), desc=pbar_desc)

        for uid in index:
            yield self.reg.loc[uid], self.data[uid]

    def extract_df(self, func, show_pbar=True, **kwargs) -> pd.DataFrame:
        """
        Extract one pd.Series or pd.DataFrame for each item and
        concatenate them all along axis=1
        """
        results = {
            tuple(key): func(key, data, **kwargs)
            for key, data in self.items(show_pbar=show_pbar)
        }

        df = pd.concat(
            results,
            axis=1,
        )

        names = list(self.reg.columns) + df.columns.names[len(self.reg.columns):]
        df.rename_axis(columns=names, inplace=True)

        return df

    def iterby(self, by, show_pbar=True):
        groups = self.reg.groupby(by).groups.items()

        if show_pbar:
            groups = tqdm(groups, total=len(groups), desc=by)

        for k, uids in groups:
            subset = self.sel_mask(uids, simplify=False)
            subset.reg.drop(by, axis=1, inplace=True)
            yield k, subset

    @classmethod
    def concat(cls, stackset_dict, names):

        merged_reg = {}

        merged_data = {}

        global_id = 0

        for key, s in stackset_dict.items():

            merged_reg[key] = s.reg.rename_axis(index='local_id')

            for local_id, data in s.data.items():
                merged_data[global_id] = data
                global_id += 1

        merged_reg = pd.concat(merged_reg, names=names)

        merged_reg = merged_reg.reset_index().drop(columns='local_id')

        return cls(merged_reg, merged_data)


def stackup(stacks_dict: dict, dim_name: str) -> Stack:
    """
    stack multiple stacks along a new, outer, dimension

    the new dimension will be the last one

    :param stacks_dict:
    :param dim_name:
    :return:
    """
    assert len(stacks_dict) > 0

    # noinspection PyTypeChecker
    data: xr.DataArray = xr.concat(
        [s.data for s in stacks_dict.values()],
        pd.Index(list(stacks_dict.keys()), name=dim_name),
    )

    stack = Stack(data)

    # TODO fix order of coords not matching. This is a workaround
    return stack.transpose().transpose()


def stackup_multidim(data: list, reg: pd.DataFrame):
    """
    Stack multiple stacks along multiple dimensions

    :param data: a list of stacks
    :param reg: a pd.DataFrame or compatible structure indicating,
        for each stack, the coords values along multiple dimensions (columns).
    :return:
    """
    reg = pd.DataFrame(reg)
    assert len(reg) == len(data)

    multi_index = pd.MultiIndex.from_frame(reg)
    multi_index.name = 'concat_dim'

    full = xr.concat([entry.data for entry in data], dim=multi_index)
    # noinspection PyTypeChecker
    full = full.unstack(multi_index.name)

    return Stack(full)


def _equal_coords(coords0, coords1):
    """check two sets of coords match exactly"""
    if coords0.keys() != coords1.keys():
        return False

    for k in coords0.keys():
        if not np.allclose(coords0[k].values, coords1[k].values):
            return False

    return True


def _equal_coords_all(coords_list):
    """check all sets of coords match exactly"""

    if len(coords_list) <= 1:
        return True

    return all(
        _equal_coords(coords_list[0], c)
        for c in coords_list
    )


def concat(stacks_list, dim='time'):
    """
    concatenate multiple stacks along an existing dimension
    :param stacks_list:
    :param dim:
    :return:
    """
    assert len(stacks_list) >= 1
    if len(stacks_list) == 1:
        return stacks_list

    assert _equal_coords_all([s.get_coords_except(dim) for s in stacks_list]), \
        f'All coords but {dim} should match'

    ref = stacks_list[0]
    values = np.concatenate([s.values for s in stacks_list], axis=ref.get_axis_num(dim))

    new_coord = [s.coords[dim].values for s in stacks_list]
    new_coord = np.concatenate(new_coord)

    coords = {
        k: (v.values if k != dim else new_coord)
        for k, v in ref.coords.items()
    }

    return Stack.from_array(
        values,
        coords,
    )
