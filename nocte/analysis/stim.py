"""
Code to analyse beta under single pulses of light stimulation.
"""

import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nocte import timeslice
from nocte.analysis import video as vid
from nocte.timeslice import ms, Win


def label_pulses(win_lights, dark=False, max_length=ms(minutes=2), valid_win=None):
    win_lights = win_lights.copy()

    is_short = win_lights.lengths() < max_length

    if valid_win is not None:
        is_in_valid_win = valid_win.contains(win_lights['ref'])
    else:
        is_in_valid_win = pd.Series(np.ones(len(win_lights.index), dtype=bool), index=win_lights.index)

    is_right_type = (win_lights['cat'] == 'off') == dark

    is_pulse = (
            is_short &
            is_in_valid_win &
            is_right_type
    )

    win_lights['is_pulse'] = is_pulse

    pulses = win_lights.sel(is_pulse=True).sort_values('start')

    if len(pulses) >= 1:
        assert pulses['cat'].nunique() == 1, pulses['cat'].value_counts()

    win_lights['pulse_idx'] = pd.Series(np.arange(len(pulses)), index=pulses.index)

    return win_lights


def load_light_wins(reg, exp_name, max_pulse_length=ms(minutes=2), col='lights'):
    encoded = reg[col].fillna('').loc[exp_name]

    if encoded == '':
        return None

    else:
        light_wins = decode_light_wins(encoded)

        light_wins = label_pulses(
            light_wins,
            dark=reg.loc[exp_name, 'stim'] == 'dark pulses',
            valid_win=Win(ms(hours=2), ms(hours=11)) if reg.loc[exp_name, 'state'] == 'sleep' else None,
            max_length=max_pulse_length,
        )

        assert light_wins.are_exclusive(), exp_name
        assert light_wins.are_tight(), exp_name

        return light_wins


def decode_light_wins(encoded, start_on=True):
    times = [0.] + [float(t) for t in encoded.strip('" ').split(':')]

    wins = timeslice.Windows.build_between(np.array(times))

    mapping = {0: 'on', 1: 'off'} if start_on else {1: 'on', 0: 'off'}
    wins['cat'] = (wins.index % 2).map(mapping)

    return wins


def encode_light_wins(wins, start_on=True):
    times = wins.get_breaks().astype(int)

    assert wins['cat'].values[0] == ('on' if start_on else 'off')
    # assert times[0] == 0
    times = times[1:]

    times = ':'.join([str(t) for t in times])

    return times


def load_light_wins_multi(reg, only_pulse=False, max_pulse_length=ms(minutes=2)):
    exp_light_wins = {}

    for exp_name in tqdm(reg.experiment_names):
        light_wins = load_light_wins(reg, exp_name, max_pulse_length=max_pulse_length)

        if light_wins is not None:

            if only_pulse:
                light_wins = light_wins.sel(is_pulse=True)
                light_wins.wins.index = light_wins['pulse_idx'].astype(int)

            if len(light_wins) > 0:
                exp_light_wins[exp_name] = light_wins.copy()

    return exp_light_wins


def extract_all_exp_luminance(reg, vid_paths, override=False):
    raw_key = 'hs'
    fix_key = 'hs_fix'

    to_extract = []

    for exp_name, red_vid in tqdm(vid_paths.items(), desc='find', total=len(vid_paths)):
        lum_path = reg.get_entry(exp_name).get_path_luminance()

        if not red_vid.exists():
            logging.error(f'Skipping {exp_name}. No reduced video: {red_vid}')

        if not override and vid.series_exists(lum_path, fix_key):
            logging.info(f'skipping existing results file: {lum_path}')

        else:
            to_extract.append((exp_name, red_vid, lum_path))

    for exp_name, red_vid, lum_path in tqdm(to_extract, desc='fix'):

        if not vid.series_exists(lum_path, raw_key):
            vid.extract_exp_luminance(red_vid, lum_path, key=raw_key)

        try:
            # noinspection PyTypeChecker
            lum_raw: pd.Series = pd.read_hdf(lum_path, raw_key)

            fixed = vid.fix_exp_luminance(
                reg.get_entry(exp_name), lum_raw,
            )

            fixed.store_hdf(str(lum_path), key=fix_key)

        except (FileNotFoundError, AssertionError) as e:
            logging.error(f'{exp_name}: {e}')


def load_all_lum_traces(reg):
    exp_lum = {}

    for exp_name in tqdm(reg.experiment_names):
        lum_path = reg.get_entry(exp_name).get_path_luminance()

        try:
            # noinspection PyTypeChecker
            lum = None

            for which in ['hs_fix', 'hs']:
                if vid.series_exists(lum_path, which):
                    # noinspection PyTypeChecker
                    lum: pd.Series = pd.read_hdf(lum_path, which)
                    break

            if lum is None:
                raise KeyError('Neither hs_fix nor hs')

            else:
                exp_lum[exp_name] = lum

        except KeyError as e:
            logging.error(f'{exp_name}: {lum_path}: {e}')

        except FileNotFoundError as e:
            logging.error(f'{exp_name}: {lum_path}: {e}')

    return exp_lum


def _label_manually(all_pulses, to_label, quiet=True):
    to_label = pd.DataFrame(to_label, columns=['exp_name', 'pulse_idx'])

    isit = pd.Series(np.zeros(len(all_pulses.index), dtype=bool), index=all_pulses.index)

    for _, exp_name, pulse_idx in to_label.itertuples():
        which = all_pulses.sel(exp_name=exp_name, pulse_idx=pulse_idx).index
        if len(which) == 0:
            if not quiet:
                logging.warning(f'Cannot find: {exp_name} pulse {pulse_idx:g}')
        else:
            assert len(which) == 1, len(which)
            which = which[0]
            isit[which] = True

    return isit


NOISY_PULSES = [
    ('GL1353_20230625_sleep', 9),
    ('GL1353_20230625_sleep', 10),
    ('GL1353_20230710_sleep', 10),
    ('GL1353_20230710_sleep', 11),
    ('GL1380_20230723_sleep', 5),
    ('GL1380_20230723_sleep', 7),
    ('GL1380_20230726_sleep', 0),
    ('GL1380_20230726_sleep', 1),
    ('GL1380_20230726_sleep', 2),
    ('GL1380_20230726_sleep', 3),
    ('GL1380_20230809_sleep', 0),
    ('GL1380_20230809_sleep', 1),
    ('GL1380_20230831_sleep', 0),
    ('GL1380_20230831_sleep', 1),
    ('GL1380_20230831_sleep', 4),
    ('GL1380_20230831_sleep', 5),
    ('GL1380_20230831_sleep', 6),
    ('GL1380_20230831_sleep', 7),
    ('GL1380_20230831_sleep', 8),
    ('GL1380_20230831_sleep', 10),
    ('GL1380_20230901_sleep', 0),
    ('GL1380_20230901_sleep', 1),
    ('GL1380_20230901_sleep', 2),
    ('GL1380_20230901_sleep', 3),
    ('GL1380_20230901_sleep', 4),
    ('GL1380_20230901_sleep', 5),
    ('GL1380_20230901_sleep', 6),
    ('GL1380_20230901_sleep', 7),
    ('GL1380_20230901_sleep', 8),
    ('GL1380_20230901_sleep', 9),
    ('GL1380_20230902_sleep', 0),
    ('GL1380_20230902_sleep', 1),
    ('GL1380_20230902_sleep', 2),
    ('GL1380_20230902_sleep', 3),
    ('GL1380_20230902_sleep', 4),
    ('GL1380_20230902_sleep', 5),
    ('GL1380_20230902_sleep', 6),
    ('GL1380_20230902_sleep', 7),
    ('GL1380_20230902_sleep', 8),
    ('GL1380_20230902_sleep', 9),
    ('GL1380_20230902_sleep', 10),
    ('GL1576_20230922_sleep', 2),
    ('GL1576_20230922_sleep', 7),
    ('GL1576_20230923_sleep', 0),
    ('GL1576_20230923_sleep', 8),
    ('GL1619_20231027_sleep', 11),
    ('GL1619_20231101_sleep', 0),
    ('GL1619_20231101_sleep', 8),
    ('GL1619_20231101_sleep', 9),
    ('GL1619_20231101_sleep', 10),
    ('GL1619_20231101_sleep', 11),
    ('GL1619_20231103_sleep', 11),
    ('GL1630_20240422_sleep', 0),
    ('GL1630_20240422_sleep', 1),
    ('GL1630_20240423_sleep', 1),
    ('GL1630_20240423_sleep', 2),
    ('GL1630_20240423_sleep', 3),
    ('GL1630_20240423_sleep', 4),
    ('GL1630_20240423_sleep', 5),
    ('GL1630_20240423_sleep', 6),
    ('GL1630_20240423_sleep', 8),
    ('GL1630_20240423_sleep', 9),
    ('GL1630_20240424_sleep', 1),
    ('GL1630_20240424_sleep', 2),
    ('GL1630_20240428_sleep', 0),
    ('GL1630_20240428_sleep', 1),

    ('GL1576_20230921_sleep', 0),
    ('GL1576_20230921_sleep', 1),
    ('GL1576_20230921_sleep', 8),
    ('GL1576_20230921_sleep', 10),
    ('GL1576_20230921_sleep', 11),
]


def label_noisy(all_pulses):
    return _label_manually(all_pulses, to_label=NOISY_PULSES)


def label_decoupled(all_pulses):
    return _label_manually(
        all_pulses,
        to_label=[
            ('GL1572_20240326_sleep', 2),
            ('GL1572_20240326_sleep', 6),
            ('GL1576_20230920_sleep', 9),
            ('GL1380_20230809_sleep', 5),
            ('GL1619_20231101_sleep', 4),
            ('GL1576_20230923_sleep', 5),
        ])


def match_protocol_length(lengths):
    expected_pulse_lengths = np.concatenate([
        np.array([5, 10, 25, 50, 100, 500, 1_000]),
        [
            5_000,
            10_000,
            15_000,
            20_000,
            30_000,
            45_000,
            50_000,
            60_000,
            70_000,
            80_000,
            90_000,
            100_000,
            120_000,
            140_000,
            160_000,
            180_000,
            200_000,
            220_000,
            900_000,
            910_000,
            920_000,
            930_000,
            940_000,
            950_000,
            960_000,
            1800_000,
        ],
        np.arange(ms(minutes=2), ms(minutes=10), ms(minutes=1))[1:]
    ])

    tdiffs = lengths.values[:, np.newaxis] - expected_pulse_lengths[np.newaxis, :]
    closest = expected_pulse_lengths[np.argmin(np.abs(tdiffs), axis=1)]

    return pd.Series(closest, index=lengths.index)


def collect_all_pulses(
        exp_light_wins,
        isolation=ms(minutes=15),
        pulse_len=ms(seconds=1),
        cat='on',
        length_precision=ms(minutes=1),
        no_edges=True,
) -> timeslice.Windows:
    all_pulses = {}

    for exp_name, wins in exp_light_wins.items():
        lights = wins.sel(cat=cat).copy()

        if len(lights) == 0:
            continue

        lights['to_prev'] = lights.interval_to_prev()
        lights['to_next'] = lights.interval_to_next()

        mask = lights.is_isolated(isolation)

        if no_edges:
            if mask.index[0] == wins.index[0]:
                mask.iloc[0] = False

            if mask.index[-1] == wins.index[-1]:
                mask.iloc[-1] = False

        lengths_precise = lights.lengths()

        lengths = match_protocol_length(lengths_precise)

        mask = mask & ((lengths - lengths_precise).abs() < length_precision)

        if isinstance(pulse_len, (float, int)):
            mask = mask & (lengths == pulse_len)

        elif isinstance(pulse_len, tuple):
            mask = mask & lengths.between(*pulse_len)

        # else:
        #     logging.warning(f'Not selecting by pulse length')

        pulses = lights.sel_mask(mask).copy()
        pulses['pulse_len'] = lengths[mask]
        pulses['pulse_len_precise'] = lengths_precise[mask]

        all_pulses[exp_name] = pulses

    all_pulses = timeslice.Windows.concat(all_pulses, cycle_name='exp_name')

    return all_pulses


def collect_analysis_windows(
        reg_sel,
        align_to='stop',
        win_len=ms(minutes=30),
        quiet=False,
        **pulse_sel_kwargs,
) -> timeslice.Windows:
    exp_light_wins = load_light_wins_multi(reg_sel)

    all_pulses = collect_all_pulses(exp_light_wins, **pulse_sel_kwargs)

    if isinstance(win_len, (float, int)):
        win = Win.build_centered(0, win_len)
    else:
        assert isinstance(win_len, (tuple, Win))
        win = win_len

    analysis_windows = all_pulses.around(win, align_to, old='pulse')

    analysis_windows['aligned_to'] = align_to

    analysis_windows['noisy'] = label_noisy(analysis_windows)
    analysis_windows['decoupled'] = label_decoupled(analysis_windows)

    reg_sel = reg_sel.sel_mask(analysis_windows['exp_name'].unique())

    analysis_windows = timeslice.Windows(
        pd.merge(
            analysis_windows.wins,
            reg_sel.reg,
            left_on='exp_name',
            right_index=True,
            how='left',
        ))

    if not quiet:
        print(f'Found {len(all_pulses):,g} pulses in total')

    return analysis_windows
