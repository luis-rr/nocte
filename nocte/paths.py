"""
Locate files (raw data and partial results).
All information is in an excel sheet that we process here (the Registry).
All partial results and processed data are stored in HDF5 files within a folder "swsort" next to the data.
"""
import datetime
import functools
import itertools
import logging
import os
from pathlib import Path
from socket import gethostname

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as pbar

import nocte.traces
from nocte import timeslice
from nocte.stacks import Stack, StackSet


def get_root():
    """path to collaboration data folder"""

    if gethostname() == 'luis-xps':
        # path to local test data
        root = Path('/home/riquelmej/dev/nocte/data/interim')

    else:
        # path in gpfs
        if os.name == 'nt':
            gpfs_loc = Path('//gpfs.corp.brain.mpg.de')
        else:
            gpfs_loc = Path('/gpfs')

        # root = Path(gpfs_loc / 'laur/collaboration/lorenz_spikes')
        root = Path(gpfs_loc / 'laur/data/fenkl')

    return root


class Entry:
    def __init__(self, reg, exp_name):
        self._reg = reg
        self.name = exp_name

    @staticmethod
    def _simplify_probe_cols(reg):
        probe_info = {}

        for idx, row in reg.iterrows():
            k = row['probe_idx']
            row = row[['probe_idx', f'probe{k}', f'side{k}', f'ch{k}']].copy()
            row.index = ['ch', 'area', 'side', 'channel']
            row['ch'] = f'ch{k}'
            probe_info[idx] = row

        probe_info = pd.DataFrame.from_dict(probe_info, orient='index')

        # noinspection PyTypeChecker
        probe_columns = list(np.concatenate([
            [f'probe{k}', f'side{k}', f'ch{k}'] for k in range(4)
        ]))

        probe_info = pd.concat([
            reg.drop(probe_columns, axis=1),
            probe_info,
        ], axis=1)

        return probe_info

    def __getitem__(self, item):
        return self._reg.loc[self.name, item]

    def get_events(self, event_cols=('light off', 'light on', 'sleep on', 'sleep off')) -> pd.Series:
        """get timestamps for all imporant events
        (like turning lights off) for this epxeriment """

        event_cols = list(event_cols)

        entries = self[event_cols].dropna()

        return entries.map(
            lambda v: datetime.timedelta(
                hours=v.hour,
                minutes=v.minute,
                seconds=v.second,
                microseconds=v.microsecond
            ).total_seconds() * 1_000,
        )

    def get_path(self, col='raw'):
        """
        get full path to a part of the experiment
        """

        if col not in self._reg.columns and f'{col}_path' in self._reg.columns:
            col = f'{col}_path'

        if col not in self._reg.columns:
            raise KeyError('Col must be one of: ' + ', '.join([c for c in self._reg.columns if c.endswith('_path')]))

        path = Path(self[col])

        if not path.is_absolute():
            path = get_root() / path

        if not path.exists():
            logging.warning(f"Path doesn't exist: {path}")

        return path

    def get_probe_channels(self, area='') -> list:
        mask = self[['probe0', 'probe1', 'probe2', 'probe3']].fillna('').str.contains(area)
        mask = mask.values

        return [
            (int(name[-1]), int(idx))
            for name, idx in self[['ch0', 'ch1', 'ch2', 'ch3']][mask].dropna().items()
        ]

    def get_loader(self, accept_non_interp=False):
        """
        Get a loader of raw data. Probe-dependent.
        """

        probe = self['probe']

        if probe == 'neuropixel':
            from nocte import io_neuropixel
            try:
                raw = io_neuropixel.MultiProbeLoader.multiprobe_interp(self.get_path('raw'))
            except FileNotFoundError:
                if not accept_non_interp:
                    raise
                else:
                    raw = io_neuropixel.MultiProbeLoader.multiprobe_spikeglx(self.get_path('raw'))

        else:
            assert probe in ['neuronexus', 'CamNeurotech'], f'Unknown probe {probe}'

            from nocte import io_neuralynx
            raw = io_neuralynx.MultiNCSLoader.from_folder(self.get_path('raw'))

        return raw

    def get_loader_simplified(self, area='', accept_non_interp=False):
        loader = self.get_loader(accept_non_interp=accept_non_interp)
        load_chans = loader.channel_probes_to_global(self.get_probe_channels(area))
        loader = loader.sel_channels(load_chans)
        assert np.all(loader.channels.index.values == np.arange(len(loader.channels.index)))

        return loader

    def get_path_video(self, cam='cam0'):
        vid_path = None

        folder = self.get_path() / 'video'

        if not folder.exists():
            folder = self.get_path() / 'videos'

        if not folder.exists():
            logging.error(f'Video folder missing. Expected: {folder}')

        else:
            try:
                vid_paths = list(folder.glob(f'{cam}_20*.avi'))

                if len(vid_paths) == 0:
                    vid_paths = list(folder.glob('cam*_20*.avi'))
                    if len(vid_paths) == 0:
                        logging.warning(f'Missing video for {self.name}. Expected in: {folder}')

                if len(vid_paths) > 1:
                    logging.warning(f'Found {len(vid_paths)} videos for {self.name}')

                    shortest = vid_paths[0]
                    for path in vid_paths:
                        if len(str(path)) < len(str(shortest)):
                            shortest = path

                    vid_paths = [shortest]

                assert len(vid_paths) == 1, f'Failed to find single video: {vid_paths}'
                vid_path = vid_paths[0]

            except PermissionError as e:
                logging.error(f'{self.name}: {e}')

        return vid_path

    def get_path_video_reduced(self, cam='cam0'):
        path = self.get_path_video(cam)
        return Path(f'{path.parent}/{path.name[:5]}reduced.avi')

    def get_path_power(
            self, band: str, probe: int, ch: int, sliding_win: int, sliding_step: int,
            suffix=''
    ) -> Path:

        location = f'p{int(probe)}c{int(ch)}'
        params = f'w{sliding_win:g}_s{sliding_step:g}'
        return self.get_path() / f'swsort/power_{band}_{location}_{params}{suffix}.h5'

    def get_path_luminance(self):
        return self.get_path() / 'swsort' / 'luminance.h5'

    def load_all_beta(
            self,
            band='beta',
            sliding_win=10_000,
            sliding_step=1_000,
            area='',
            simplify=True,
    ) -> pd.DataFrame:

        df = pd.DataFrame({
            (p, c): pd.read_hdf(
                self.get_path_power(
                    band=band, probe=p, ch=c,
                    sliding_win=sliding_win, sliding_step=sliding_step)
            )
            for p, c in self.get_probe_channels(area=area)
        })

        assert len(df) > 0

        df.rename_axis(columns=['probe', 'channel'], inplace=True)

        if simplify:
            df = df.droplevel('channel', axis=1)
            df.columns = [col for col in df.columns]
            df.rename_axis(columns=['probe_idx'], inplace=True)

        return df

    def load_all_beta_traces(self, simplify=True, *args, **kwargs):

        loaded = self.load_all_beta(simplify=simplify, *args, **kwargs)

        traces: nocte.traces.Traces = nocte.traces.Traces.from_df(
            loaded,
        )

        traces['exp_name'] = self.name

        for k, v in self._reg.loc[self.name].items():
            traces[k] = v

        if simplify:
            traces.reg = Entry._simplify_probe_cols(traces.reg)

        traces.reg.dropna(axis=1, how='all', inplace=True)

        return traces


class Registry:
    def __init__(self, df: pd.DataFrame):
        assert df.index.is_unique, df.index[df.index.duplicated()]

        if df.index.name is None:
            df.index.name = 'name'

        valid_path = df['raw_path'].notna()
        for name in df.index[~valid_path]:
            logging.error(f'Dropping {name}: missing path.')

        df: pd.DataFrame = df.loc[df['raw_path'].notna()].copy()

        for col in df.columns[df.columns.str.endswith('_path')]:
            df[col] = Registry._patch_paths(df[col])

        # noinspection PyUnresolvedReferences
        assert (df.groupby('animal')['lesion'].nunique() <= 1).all()

        df = df.replace('?', np.nan)
        self.reg = df.copy()

        paths = pd.Series({
            name: self.get_path(name) for name in self.experiment_names
        }, dtype=object)

        valid_path = paths.map(lambda p: p.exists())
        for name in self.reg.index[~valid_path]:
            logging.error(f'Dropping {name}: path does not exist ({paths[name]})')

        self.reg = self.reg.loc[df['raw_path'].notna()].copy()

        invalid = self._detect_invalid_probe_info()
        if invalid.any():
            logging.warning(f'Missing information for probes in {invalid.index[invalid]}')
            self.reg = self.reg[~invalid]

        probe_counts = pd.Series({
            exp_name: len(self.get_probe_channels(exp_name))
            for exp_name in self.experiment_names
        }, dtype=object)

        probe_counts_totals = probe_counts.value_counts()

        if probe_counts_totals.get(0, default=0) > 0:
            logging.warning(
                f'{probe_counts_totals[0]} experiments without probes! '
                f'Dropping: {list(probe_counts.index[probe_counts == 0])}'
            )
            # noinspection PyUnresolvedReferences
            self.reg = self.sel_mask(probe_counts != 0).reg

    def copy(self):
        return self.__class__(self.reg.copy())

    def _detect_invalid_probe_info(self) -> pd.Series:
        probe_info = self.reg[[
            'probe0', 'probe1', 'probe2', 'probe3',
            'side0', 'side1', 'side2', 'side3',
            'ch0', 'ch1', 'ch2', 'ch3',
        ]].copy()

        probe_info.columns = pd.MultiIndex.from_product([['probe', 'side', 'ch'], [0, 1, 2, 3]], names=['which', 'idx'])

        invalid = probe_info.isna().T.groupby('idx').any().T

        return invalid.all(axis=1)

    @staticmethod
    def _patch_paths(paths):

        patched = paths.dropna().astype(str)
        patched = patched.str.replace('\\\\gpfs.corp.brain.mpg.de', '\\gpfs', regex=False)
        patched = patched.str.replace('\\', '/', regex=False)
        paths_exist = patched.map(lambda p: Path(p).exists()).reindex(patched.index, fill_value=False)
        alt = patched.str.replace('/gpfs/laur/experiments/FenkLorenz', '/gpfs/laur/data/fenkl')
        alt_exist = alt.dropna().map(lambda p: Path(p).exists()).reindex(alt.index, fill_value=False)
        to_patch = (~paths_exist) & alt_exist

        if to_patch.any():
            logging.warning(f'Patching {np.count_nonzero(to_patch)} paths: {list(to_patch.index[to_patch])}')
            patched.loc[to_patch] = alt.loc[to_patch]

        patched = patched.reindex(paths.index)

        return patched

    @classmethod
    def read_excel(cls, reg_path=None, sheet_name='swr'):
        """load a stored registry of all of the experiments and important paths"""

        if reg_path is None:
            reg_path = get_root() / 'spikes/registry_merged.xlsx'

        # noinspection PyTypeChecker
        reg = pd.read_excel(reg_path, index_col='name', sheet_name=sheet_name)

        df = reg.dropna(how='all')

        to_ignore = df['ignore'].fillna(False).astype(bool)
        df = df[~to_ignore]

        return cls(df)

    def sel_mask(self, mask, but=False):
        """
        Select using a boolean mask
        """
        if but:
            mask = ~mask

        return self.__class__(self.loc[mask])

    def sel_masks(self, criterias, *, how='all', but=False):
        """
        Select using multiple boolean masks to be combined
        """
        assert how in ('all', 'any')

        if how == 'all':
            mask = np.all(criterias, axis=0)
        else:
            mask = np.any(criterias, axis=0)

        return self.sel_mask(mask, but=but)

    def sel(self, *, how='all', but=False, **col_values):
        """
        Select by direct comparison of some column.
        For example:
            wins.sel(cat='baseline')
        """
        criterias = [
            (self[col] == value)
            for col, value in col_values.items()
        ]

        return self.sel_masks(criterias, how=how, but=but)

    def sel_between(self, *, but=False, how='all', **col_ranges):
        """
        Select by direct comparison of some column where values in a range are acceptable.
        For example:
            wins.sel_between(duration=(0, 60_000))
        """
        criterias = [
            self[col].between(*value_range)
            for col, value_range in col_ranges.items()
        ]

        return self.sel_masks(criterias, how=how, but=but)

    def sel_isin(self, *, but=False, how='all', **col_values):
        """
        Select by direct comparison of some column where any of the values are acceptable.
        For example:
            wins.sel_isin(cat=['sws', 'rem'])
        """
        criterias = [
            self[col].isin(values)
            for col, values in col_values.items()
        ]

        return self.sel_masks(criterias, how=how, but=but)

    @functools.wraps(pd.DataFrame.__getitem__)
    def __getitem__(self, *args, **kwargs):
        return self.reg.__getitem__(*args, **kwargs)

    @functools.wraps(pd.DataFrame.__setitem__)
    def __setitem__(self, *args, **kwargs):
        return self.reg.__setitem__(*args, **kwargs)

    def is_bilat(self, area: str) -> pd.Series:
        count = np.zeros(len(self.reg))
        for col in ['probe0', 'probe1', 'probe2', 'probe3']:
            count = count + (self[col].str.lower() == area.lower()).astype(int)

        return count >= 2

    def is_stim(self):
        return self['stim'].notna()

    def is_sleep(self):
        return self['state'] == 'sleep'

    def is_healthy(self) -> pd.Series:
        return self['lesion'].isna()

    def is_lesion(self, which: str) -> pd.Series:
        return self['lesion'].str.lower().str.contains(which.lower()).fillna(False)

    def __len__(self):
        return len(self.reg)

    def get_entry(self, exp_name):
        return Entry(self, exp_name)

    def get_events(self, exp_name, *args, **kwargs) -> pd.Series:
        return self.get_entry(exp_name).get_events(*args, **kwargs)

    def get_all_events(self) -> pd.DataFrame:
        """get timestamps for all imporant events
        (like turning lights off) for all epxeriments """
        return pd.DataFrame({
            exp_name: self.get_events(exp_name)
            for exp_name in self.experiment_names
        })

    def get_path(self, exp_name, *args, **kwargs):
        return self.get_entry(exp_name).get_path(*args, **kwargs)

    def get_path_sne(self, exp_name, probe: int, ch: int, suffix='') -> Path:
        return self.get_path(exp_name) / f'swsort/sne_p{probe}c{ch}{suffix}.h5'

    def get_path_sne_all(self, exp_name, area='CLA', suffix='') -> list:
        paths = []

        for i, (p, ch) in enumerate(self.get_probe_channels(exp_name, area=area)):
            # noinspection PyTypeChecker
            paths.append((p, ch, self.get_path_sne(exp_name, p, ch, suffix=suffix)))

        return paths

    def get_path_matching(
            self, exp_name,
            suffix='', area=''
    ) -> Path:
        pcs = self.get_probe_channels(exp_name, area=area)
        assert len(pcs) >= 2
        return (
                self.get_path(exp_name) /
                f'swsort/sne_matching_p{pcs[0][0]}c{pcs[0][1]}_p{pcs[1][0]}c{pcs[1][1]}{suffix}.h5'
        )

    def get_path_power(self, exp_name, *args, **kwargs) -> Path:
        return self.get_entry(exp_name).get_path_power(*args, **kwargs)

    def get_path_xcorr(
            self, exp_name, probe0: int, probe1: int, ch0: int, ch1: int, sliding_win: int,
            low_hz=40, suffix='',
    ) -> Path:
        location = f'p{int(probe0)}c{int(ch0)}_p{int(probe1)}c{int(ch1)}'
        params = f'w{sliding_win:g}'
        return self.get_path(exp_name) / f'swsort/xcorr_{location}_{params}{suffix}_{low_hz:g}hz.h5'

    def get_path_xcorr_area(
            self, exp_name, sliding_win: int,
            low_hz=40, suffix='', area=''):
        pcs = self.get_probe_channels(exp_name, area=area)

        return self.get_path_xcorr(
            exp_name,
            pcs[0][0], pcs[1][0], pcs[0][1], pcs[1][1],
            sliding_win=sliding_win, low_hz=low_hz, suffix=suffix,
        )

    def collect_paths_power(self, band, sliding_win, sliding_step, areas=None, missing=False):
        to_extract = []

        for exp_name in self.experiment_names:

            probe_idcs = self._get_valid_probe_idcs(exp_name, areas=areas)

            for idx in probe_idcs:

                ch = self.reg.loc[exp_name, f'ch{idx}']
                if np.isnan(ch):
                    logging.error(f'Missing channel {idx} for {exp_name}')
                    continue

                ch = int(ch)

                results_path = self.get_path_power(exp_name, band, idx, ch, sliding_win, sliding_step)

                if (not results_path.exists()) == missing:
                    to_extract.append((exp_name, results_path, idx, ch))

        return pd.DataFrame.from_records(
            to_extract,
            columns=['exp_name', 'path', 'probe', 'channel'],
        )

    def collect_paths_xcorr(self, sliding_win, suffix='', missing=False, areas=None, low_hz=40) -> pd.DataFrame:
        """
        :param sliding_win:
        :param suffix:
        :param missing:
        :param low_hz:
        :param areas: List of valid areas to compute x-corr. If none specified, any area is valid.
        :return:
        """
        to_extract = []

        for exp_name in self.experiment_names:

            probe_idcs = self._get_valid_probe_idcs(exp_name, areas=areas)

            import itertools
            for a, b in itertools.combinations(probe_idcs, 2):

                ch_a = self.reg.loc[exp_name, f'ch{a}']
                ch_b = self.reg.loc[exp_name, f'ch{b}']

                if not np.isnan(ch_a) and not np.isnan(ch_b):
                    ch_a = int(ch_a)
                    ch_b = int(ch_b)

                    results_path = self.get_path_xcorr(
                        exp_name, a, b, ch_a, ch_b, sliding_win,
                        suffix=suffix, low_hz=low_hz)

                    if (not results_path.exists()) == missing:
                        to_extract.append((exp_name, results_path, a, b, ch_a, ch_b))

        return pd.DataFrame.from_records(
            to_extract,
            columns=['exp_name', 'path', 'p0', 'p1', 'ch0', 'ch1'],
        )

    def collect_paths_sne(self, missing=False, areas=None, suffix='') -> pd.DataFrame:
        to_extract = []

        for exp_name in self.experiment_names:

            probe_idcs = self._get_valid_probe_idcs(exp_name, areas=areas)

            for idx in probe_idcs:

                if self.loc[exp_name, f'probe{idx}'] in ['CLA', 'BST']:

                    ch = int(self.reg.loc[exp_name, f'ch{idx}'])

                    results_path = self.get_path_sne(exp_name, idx, ch, suffix=suffix)

                    if (not results_path.exists()) == missing:
                        to_extract.append((exp_name, results_path, idx, ch))

        return pd.DataFrame.from_records(
            to_extract,
            columns=['exp_name', 'path', 'probe', 'channel'],
        )

    def collect_paths_matching(self, area='', missing=False, suffix='') -> pd.DataFrame:

        paths = []

        for exp_name in self.experiment_names:
            results_path = self.get_path_matching(exp_name, area=area, suffix=suffix)

            if (not results_path.exists()) == missing:
                paths.append((exp_name, results_path))

        return pd.DataFrame.from_records(
            paths,
            columns=['exp_name', 'path']
        )

    def _collect_paths_glob(self, pattern):
        exp_paths = {}

        for exp_name in pbar(self.experiment_names, desc='find'):
            paths = [str(p) for p in self.get_path(exp_name).glob(pattern)]

            if len(paths) > 0:
                if len(paths) == 1:
                    exp_paths[exp_name] = paths[0]

                else:

                    shortest = min(paths, key=len)

                    logging.warning(
                        f'Exp {exp_name} has {len(paths)} files:\n' + '\n'.join(paths) + f'\nTaking:\n{shortest}')

                    exp_paths[exp_name] = shortest

        return pd.Series(exp_paths)

    def collect_paths_jrclust(self):
        return self._collect_paths_glob('binaryTest*.csv')

    def collect_paths_deeplabcut(self):
        return self._collect_paths_glob('DeepLabCut/*.csv')

    def _get_valid_probe_idcs(self, exp_name, areas=None):
        probes = self.reg.loc[exp_name, ['probe0', 'probe1', 'probe2', 'probe3']].dropna()

        if areas is not None:
            probes = probes[probes.isin(areas)]

        probe_idcs = [int(p[-1]) for p in probes.index]

        return probe_idcs

    # noinspection PyTypeChecker
    @property
    @functools.wraps(pd.DataFrame.loc)
    def loc(self):
        return self.reg.loc

    @functools.wraps(pd.DataFrame.value_counts)
    def value_counts(self, *args, **kwargs):
        return self.reg.value_counts(*args, **kwargs)

    @property
    def index(self) -> pd.Index:
        """pd.DataFrame accessor"""
        return self.reg.index

    @property
    def columns(self) -> pd.Index:
        """pd.DataFrame accessor"""
        return self.reg.columns

    @property
    def experiment_names(self):
        return self.reg.index

    def _repr_html_(self):
        """pretty print on notebooks"""
        # noinspection PyProtectedMember
        return self.reg._repr_html_()

    def group_exps(self, names=None, by=('state', 'lesion'), count_label=True) -> (pd.Series, pd.DataFrame):
        """
        Groups experiments by multiple columns and creates a unique style for each group.
        Style includes "color" and "label".
        """
        if names is None:
            names = self.experiment_names

        if isinstance(by, str):
            by = [by]

        by = list(by)

        entries = self.reg.loc[names].fillna('none').copy()
        # allow grouping by the name
        entries = entries[pd.Index(by).difference(['name'])]

        group_sizes = entries.reset_index(drop=False).groupby(by).size().sort_values(ascending=False)
        groups_ids = pd.Series(np.arange(len(group_sizes)), group_sizes.index, name='group_id')
        groups: pd.DataFrame = groups_ids.reset_index().set_index('group_id')

        labels = {}

        for i, vs in groups[by].T.items():
            labels[i] = ', '.join([str(v) for v in vs if v != 'none'])
            if labels[i] == '':
                labels[i] = 'none'

            if count_label:
                labels[i] = f'{labels[i]} ({group_sizes[i]})'

        groups['label'] = pd.Series(labels)
        groups['color'] = ['xkcd:grey'] + [f'C{i}' for i in np.arange(len(groups) - 1)]

        entry_group = pd.merge(entries.rename_axis(index='name').reset_index(drop=False),
                               groups.reset_index(), how='left').set_index('name')['group_id']

        return entry_group, groups

    def iter_groupby(self, *args, pbar=None, **kwargs):

        grouped = self.reg.groupby(*args, **kwargs)

        if pbar is not None:
            grouped = pbar(list(grouped))

        for k, sreg in grouped:
            yield k, self.__class__(sreg)

    def get_exp_short_desc(self, exp_name, cols=('probe', 'lesion', 'stim', 'state')):
        """get a short string description of this experiment"""
        return ', '.join(self.reg.loc[exp_name, list(cols)].dropna())

    def get_loader(self, exp_name, *args, **kwargs):
        return self.get_entry(exp_name).get_loader(*args, **kwargs)

    def get_probe_channels(self, exp_name, *args, **kwargs) -> list:
        return self.get_entry(exp_name).get_probe_channels(*args, **kwargs)

    def get_loader_simplified(self, exp_name, *args, **kwargs):
        return self.get_entry(exp_name).get_loader_simplified(*args, **kwargs)

    def load_timestamps(self, col):

        def parse_entry(string):
            desc, time_str = string.split(' - ')

            time_ms = timeslice.timestamp_to_milliseconds(time_str)

            return desc, time_ms

        events_desc = self[col].dropna()

        table = []

        for exp_name, events_desc in events_desc.items():
            for i, entry in enumerate(events_desc.split('\n')):
                desc, time = parse_entry(entry)

                table.append((exp_name, desc, time, i))

        return pd.DataFrame(table, columns=['exp_name', 'desc', 'time', f'{col}_idx'])

    def load_wins(self, col):
        lights_desc = self[col].dropna()

        exp_wins = {
            exp_name: timeslice.Windows.from_str(lights_str)
            for exp_name, lights_str in lights_desc.items()
        }

        return exp_wins

    def load_all_sne(self, exp_name, **kwargs) -> pd.DataFrame:

        paths = self.get_path_sne_all(exp_name, **kwargs)

        all_sns = []

        for i, (p, ch, path) in enumerate(paths):
            # noinspection PyTypeChecker
            df: pd.DataFrame = pd.read_hdf(path)

            df['probe'] = p
            df['channel'] = ch

            all_sns.append(df)

        return pd.concat(all_sns, axis=0, ignore_index=True)

    def load_all_beta(self, exp_name, *args, **kwargs) -> pd.DataFrame:
        return self.get_entry(exp_name).load_all_beta(*args, **kwargs)

    def load_all_beta_traces_multi(self, *args, show_pbar=True, **kwargs):

        all_traces = []

        to_load = self.experiment_names

        if show_pbar is not False:
            show_pbar = pbar if show_pbar is True else show_pbar
            to_load = show_pbar(to_load, desc='load beta')

        for exp_name in to_load:
            traces = self.get_entry(exp_name).load_all_beta_traces(*args, **kwargs)

            traces = traces.resample(start=0, period=500)

            all_traces.append(traces)

        all_traces = nocte.traces.Traces.concat_list(all_traces)

        return all_traces

    def load_all_beta_norm(
            self, exp_name,
            simplify=True, area='CLA', exp_valid_win=None,
            add_max=True, add_mean=False,
    ):
        all_beta = self.load_all_beta(exp_name, band='beta', area=area, simplify=simplify)

        # note it is important to crop before computing quantiles
        # because motion artifacts are much more likely at the beginning/end of the recording
        if exp_valid_win is not None:
            all_beta = exp_valid_win.crop_df(all_beta)

        all_beta = all_beta / all_beta.quantile(.999)

        if add_max:
            all_beta['beta_max'] = all_beta.max(axis=1)

        if add_mean:
            all_beta['beta_mean'] = all_beta.mean(axis=1)

        return all_beta

    def load_area_beta(self, exp_name, simplify=True, **kwargs):

        traces = self.load_all_beta_norm(exp_name, simplify=False, add_max=False, **kwargs)

        areas = [self.loc[exp_name, f'probe{p}'] for p in traces.columns.get_level_values('probe')]
        sides = [self.loc[exp_name, f'side{p}'] for p in traces.columns.get_level_values('probe')]

        traces.columns = pd.MultiIndex.from_frame(pd.DataFrame({
            'area': areas,
            'side': sides,
            'channel': traces.columns.get_level_values('probe'),
        }))

        if simplify:
            cols = [f'{side}_{area}_{ch}' for area, side, ch in traces.columns]
            traces.columns = cols

        return traces

    def load_all_beta_norm_multi(self, **kwargs) -> dict:
        exp_beta = {}
        for exp_name in pbar(self.experiment_names, desc='load beta'):
            try:
                exp_beta[exp_name] = self.load_all_beta_norm(exp_name, **kwargs)

            except FileNotFoundError as e:
                logging.error(f'Missing data for {exp_name}: {e}')

        return exp_beta

    def load_exp_xcorr_combs(self, exp_names, which=None, **xcorr_kwargs) -> StackSet:

        if isinstance(which, str):
            which = [which]

        xcorr_triplet = {}
        all_paths = {}

        for exp_name in exp_names:
            probe_channels = self.get_probe_channels(exp_name)

            for (p0, c0), (p1, c1) in itertools.combinations(probe_channels, 2):

                area0 = self.loc[exp_name, f'probe{p0}']
                area1 = self.loc[exp_name, f'probe{p1}']

                side0 = self.loc[exp_name, f'side{p0}']
                side1 = self.loc[exp_name, f'side{p1}']

                swap = False

                # give preference to claustrum as our reference
                # this means we may need to swap the lags
                if area1 == 'CLA' and area0 != 'CLA':
                    swap = True
                    area0, area1 = area1, area0
                    side0, side1 = side1, side0

                if area0 == area1:
                    key = f'{area0} bilat'

                elif side0 == side1:
                    key = f'{area0}-{area1} ipsi'

                else:
                    key = f'{area0}-{area1} contra'

                if which is None or key in which:
                    path = self.get_path_xcorr(exp_name, p0, p1, c0, c1, **xcorr_kwargs)

                    if path.exists():
                        all_paths[key, exp_name] = path, swap

                    else:
                        logging.warning(f'{exp_name}: Missing expected x-corr {p0}-{c0} vs {p1}-{c1}: {path}')

        # noinspection PyTypeChecker
        for (key, exp_name), (path, swap) in pbar(all_paths.items(), desc='load x-corr'):
            print(f'loading {path}')
            xcorr = Stack.load_hdf(str(path), 'xcorr')

            if swap:
                print('swapping')
                xcorr = xcorr.replace_dim('lag', 'lag', xcorr.coords['lag'] * -1)
                xcorr = xcorr.sortby('lag')

            xcorr_triplet[key, exp_name] = xcorr

        return StackSet.from_dict(xcorr_triplet, names=['pair', 'exp'])

    def collect_paths_video(self) -> pd.Series:
        raw_video_paths = {}

        for exp_name in self.experiment_names:

            vid_path = self.get_entry(exp_name).get_path_video()

            if vid_path is not None:
                raw_video_paths[exp_name] = vid_path

        raw_video_paths = pd.Series(raw_video_paths)

        return raw_video_paths

    def collect_paths_video_reduced(self) -> pd.Series:
        raw_vid_paths = self.collect_paths_video()

        return pd.Series({
            name: Path(f'{path.parent}/{path.name[:5]}reduced.avi')
            for name, path in raw_vid_paths.items()
        })
