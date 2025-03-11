"""
Code for beta phase analysis.
"""

import numpy as np
import pandas as pd
import scipy

import nocte.traces
from nocte import stacks
from nocte.timeslice import ms


def get_hilbert_transform(
        trace,
        filter_hz=(
                1_000 / ms(minutes=6),
                1_000 / ms(seconds=60),
        ),
        phase_shift=True,
):
    period = np.diff(trace.index)
    assert np.allclose(period[0], period)
    period = period[0]

    trace = stacks.Stack.from_series(trace).filter_pass(filter_hz).to_series()

    x_a = scipy.signal.hilbert(trace.values)

    analytic = {
        'real': x_a.real,
        'imag': x_a.imag,
        'phase': np.angle(x_a),
        'power': np.abs(x_a) ** 2,
    }

    analytic['freq'] = np.append(np.nan, np.diff(analytic['phase']) / (2 * np.pi * period))

    if phase_shift:
        analytic['phase'] = (analytic['phase'] + 1.5 * np.pi) % (2 * np.pi) - np.pi

    return pd.DataFrame(analytic, index=trace.index)


def get_phase(trace, **kwargs):
    if isinstance(trace, pd.DataFrame):
        df = pd.DataFrame({
            k: get_phase(tr, **kwargs)
            for k, tr in trace.items()
        })

        df.columns = trace.columns

        return df

    hilb = get_hilbert_transform(trace, **kwargs)

    return hilb['phase']


def get_phase_norm(trace, **kwargs):
    phase = get_phase(trace, **kwargs)

    phase_norm = (phase + np.pi) / (np.pi * 2)

    return phase_norm


def collect_beta_phases(exp_beta: nocte.traces.Traces, unwrap=True, **kwargs):
    exp_phase: nocte.traces.Traces = exp_beta.apply(
        lambda beta: get_phase_norm(
            beta.dropna(),
            **kwargs
        )
    )

    if unwrap:
        exp_phase = exp_phase.drop_missing()  # unwrap doesn't support nans
        exp_phase = exp_phase.unwrap()

    return exp_phase


def classify_phase(values):
    idcs = np.digitize(values, bins=np.linspace(0, 1, 5)) - 1

    idcs = pd.Series(idcs, values.index)

    return idcs.map({
        0: 'early sws',
        1: 'late sws',
        2: 'early rem',
        3: 'late rem',
    })


def get_phase_cut(phase_traces, analysis_windows, phase_time=ms(seconds=-5), name='single'):
    phase_detailed_cut = phase_traces.extract(analysis_windows)

    phase_detailed_cut = phase_detailed_cut - np.floor(phase_detailed_cut.tloc[phase_time])

    phase_detailed_cut[f'phase_{name}'] = phase_detailed_cut.lookup(phase_time)
    phase_detailed_cut[f'phase_{name}_cat'] = classify_phase(phase_detailed_cut[f'phase_{name}'])

    return phase_detailed_cut


def take_phase_diff(phase_cut, phase_time, class_by, n_periods=1):
    actual_phase = phase_cut.lookup(phase_time + phase_cut['cycle_length'] * n_periods, interp=True)

    expected_phase = phase_cut[class_by] + n_periods

    phase_diff = actual_phase - expected_phase

    max_diff = n_periods

    shift = pd.Series(0, index=phase_diff.index)

    shift[phase_diff < -max_diff] = +1
    shift[phase_diff > +max_diff] = -1

    return phase_diff + shift
