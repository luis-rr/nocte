"""
Code to handle samples, events and collections of events
"""

import numba
import numpy as np
import pandas as pd

from nocte.events import Events


@numba.njit(parallel=True)
def _extract_cdf_other_nb(this, null):
    counts = np.empty(this.shape[0])
    for i in numba.prange(this.shape[0]):

        # noinspection PyTypeChecker
        mask: np.ndarray = (null >= this[i])

        which = np.ones(mask.shape[0], dtype=np.bool_)

        for row in mask.T:
            which = which & row

        count = np.count_nonzero(which)
        counts[i] = count

    return counts / null.shape[0]


class SharpNegativeEvents(Events):
    def __init__(self, reg: pd.DataFrame):
        super().__init__(reg)
        self.reg.rename_axis(index='event_id', inplace=True)

    @classmethod
    def from_double_acceleration(
            cls,
            main,
            band_hz=(.25, 40),
    ):
        assert len(main.dims) == 1

        filtered = main.filter_pass(band_hz)

        speed = filtered.gradient()
        acc = speed.gradient()

        neg_spe = speed.find_peaks(negative=True, height=0, prominence=0, width=0)
        neg_acc = acc.find_peaks(negative=True, height=0, prominence=0, width=0)
        pos_acc = acc.find_peaks(negative=False, height=0, prominence=0, width=0)

        all_peaks = pd.concat({
            'start': neg_acc,
            'ref': neg_spe,
            'stop': pos_acc,
        },
            axis=0, names=['point']
        )
        all_peaks.sort_values('time', inplace=True)
        all_peaks.reset_index('point', inplace=True)
        all_peaks.reset_index(drop=True, inplace=True)

        all_peaks.rename(
            columns=dict(
                sample_idx='idx',
                width_height='rel_height',
            ),
            inplace=True,
        )

        consecutive = (
                (all_peaks['point'].values[:-2] == 'start') &
                (all_peaks['point'].values[1:-1] == 'ref') &
                (all_peaks['point'].values[2:] == 'stop')
        )
        start = all_peaks.index[:-2][consecutive]
        ref = all_peaks.index[1:-1][consecutive]
        stop = all_peaks.index[2:][consecutive]

        assert len(start) == len(stop) and len(start) == len(ref)

        cols = ['time', 'idx', 'prominence', 'rel_height']

        events = pd.concat([
            all_peaks.loc[start, cols].add_prefix('start_').reset_index(drop=True),
            all_peaks.loc[ref, cols].add_prefix('ref_').reset_index(drop=True),
            all_peaks.loc[stop, cols].add_prefix('stop_').reset_index(drop=True),
        ],
            axis=1,
        )

        sns = cls(events)

        sns = sns.lookup_and_set('speed', speed.values, by='idx')
        sns = sns.lookup_and_set('acc', acc.values, by='idx')
        sns = sns.lookup_and_set('raw', main.values, by='idx')
        sns = sns.lookup_and_set('filt', filtered.values, by='idx')

        sns.reg['acc_diff'] = sns.reg['stop_acc'] - sns.reg['start_acc']
        sns.reg['amplitude_filt'] = sns.reg['stop_filt'] - sns.reg['start_filt']
        sns.reg['amplitude'] = sns.reg['stop_raw'] - sns.reg['start_raw']
        sns.reg['duration'] = sns.reg['stop_time'] - sns.reg['start_time']
        sns.reg['speed'] = sns.reg['amplitude'] / sns.reg['duration']

        return sns

    def extract_cdf_other(self, sns_null, cols=('start_acc', 'stop_acc', 'ref_speed')):

        cols = list(cols)

        rates = _extract_cdf_other_nb(
            self.reg[cols].abs().values,
            sns_null.reg[cols].abs().values,
        )

        return pd.Series(rates, self.reg.index)

    def add_details(self, all_beta):
        beta_max = all_beta.max(axis=1)
        copy = self.lookup_and_set(f'beta_max', beta_max, cols=['ref'], by='time')

        for ch, beta in all_beta.items():
            copy = copy.lookup_and_set(f'beta_ch{ch}', beta, cols=['ref'], by='time')

        copy.reg['ref_beta_local'] = pd.concat([
            copy.sel(channel=ch).reg[f'ref_beta_ch{ch}']
            for ch in all_beta.columns

        ]).sort_index()

        copy.reg['duration_log'] = np.log10(np.abs(copy.reg.loc[copy.reg['duration'] > 0, 'duration']))
        copy.reg['amplitude_log'] = np.log10(np.abs(copy.reg.loc[copy.reg['amplitude'] < 0, 'amplitude']))
        copy.reg['amplitude_filt_log'] = np.log10(np.abs(
            copy.reg.loc[copy.reg['amplitude_filt'] < 0, 'amplitude_filt']
        ))
        copy.reg['speed_log'] = np.log10(np.abs(copy.reg.loc[copy.reg['speed'] < 0, 'speed']))

        copy.reg['ref_speed_log'] = np.log10(np.abs(copy.reg.loc[copy.reg['ref_speed'] < 0, 'ref_speed']))
        copy.reg['start_acc_log'] = np.log10(np.abs(copy.reg.loc[copy.reg['start_acc'] < 0, 'start_acc']))
        copy.reg['stop_acc_log'] = np.log10(np.abs(copy.reg.loc[copy.reg['stop_acc'] > 0, 'stop_acc']))
        copy.reg['acc_diff'] = copy.reg['stop_acc'] - copy.reg['start_acc']
        copy.reg['speed_diff'] = copy.reg['stop_speed'] - copy.reg['start_speed']

        copy.reg['ref_to_stop_time'] = copy.reg['stop_time'] - copy.reg['ref_time']
        copy.reg['ref_to_start_time'] = copy.reg['start_time'] - copy.reg['ref_time']

        for col in ['acc_diff', 'ref_prominence', 'start_prominence', 'stop_prominence']:
            copy.reg[f'{col}_log'] = np.log10(np.abs(copy.reg.loc[copy.reg[col] > 0, col]))

        return copy

    def add_isi(self):
        copy = self.reg.copy()
        copy['isi_next'] = self.get_inter_event_intervals()
        copy['isi_prev'] = copy['isi_next'].shift(1)
        return self.__class__(copy)

    def assign_matches(
            self,
            path: pd.DataFrame,
            name='match',
            cols=(
                    'amplitude', 'speed', 'duration',
                    'ref_time', 'start_time',
            ),
            quiet=False
    ):
        copy = self.reg.copy()

        copy.loc[path[0], name] = path[1].values
        copy.loc[path[1], name] = path[0].values

        matches = copy[name].dropna().astype(int)

        for col in cols:
            copy.loc[matches.index, f'{name}_{col}'] = copy.loc[matches.values, col].values
            copy[f'{name}_{col}_diff'] = copy[f'{name}_{col}'] - copy[col]

        if not quiet:
            matched_count = np.count_nonzero(copy[name].notna())
            print(f'matched {matched_count:,g}/{len(copy):,g} ({100 * matched_count / len(copy):.1f}%) events')

        return self.__class__(copy)

    def get_match(self, match_col, name='match') -> pd.Series:

        matches = self.reg[name]

        missing = ~matches.dropna().isin(self.reg.index)

        if np.any(missing):
            print(f'Missing data for {np.count_nonzero(missing)}/{len(missing)} match ids')

        values = self.reg[match_col].reindex(matches.values).values

        return pd.Series(values, index=matches.index)

    def _get_simplified_channel_idx(self):
        mapping = self.reg[['probe', 'channel']].drop_duplicates().sort_values(['probe', 'channel']).copy()

        mapping_code = mapping['probe'] * 1e6 + mapping['channel']

        mapping_code = pd.Series(np.arange(len(mapping)), index=mapping_code.values)

        code = self.reg['probe'] * 1e6 + self.reg['channel']

        return code.map(mapping_code)

    def patch_simplified_channels(self):
        """
        Replace the true channel id with a consecutive ch0, ch1, ch2, etc.

        This is useful to handle different experiments where we use different channels
        due to SNR or position, but they all represent the same place.

        New channels are assigned consecutive integer values based on their
        sorted probe+channel id.

        :return: modified copy
        """

        new = self.reg.copy()
        new['channel_true'] = new['channel']
        new['channel'] = self._get_simplified_channel_idx()

        return self.__class__(new)

    @classmethod
    def load_matched_sns(cls, reg, exp_name, sn_suffix='_cdf', matching_suffix='_lax', drop_singles=False):
        """
        Return all detected SNs with bilateral matching.
        If not dropped, single (unmatched) SNs are included with nan in "match" column.
        """
        sns = cls(reg.load_all_sne(exp_name, suffix=sn_suffix))
        sns = sns.patch_simplified_channels()
        assert sns['channel'].isin([0, 1]).all()

        # noinspection PyTypeChecker
        matching: pd.DataFrame = pd.read_hdf(reg.get_path_matching(exp_name, suffix=matching_suffix))
        matched = sns.assign_matches(matching)
        matched['is_lead'] = matched['match_ref_time_diff'] > 0

        if drop_singles:
            matched = matched.sel_mask(matched['match'].notna())

        return matched

    def is_match_valid(self, match_col='match'):
        """
        Boolean mask indicating for each SN if they have a valid match.

        :param match_col:
        :return:
        """
        matches = self.reg[match_col]
        return matches.isin(self.reg.index)

    def drop_missing_matches(self, match_col='match'):
        """
        Set the match column to nan if the match is not in the registry.
        This happens sometimes when matching was done for a full experiment
        but we are looking at a slice. In that case the match of an SN may
        foll outside the slice, so we want to treat it as a single.
        """
        valid = self.is_match_valid(match_col=match_col)
        new = self.reg.copy()
        new.loc[~valid, match_col] = np.nan
        return self.__class__(new)
