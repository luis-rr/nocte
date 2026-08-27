import datetime

import numpy as np
import pytest

import nocte.core.time


def test_to_ms():
    assert nocte.core.time.to_ms(12) == 12.0
    assert nocte.core.time.to_ms(12.5) == 12.5
    assert nocte.core.time.to_ms(np.float64(12.5)) == 12.5

    delta = datetime.timedelta(seconds=1, milliseconds=250)
    assert nocte.core.time.to_ms(delta) == 1250.0


def test_ms():
    assert nocte.core.time.ms(milliseconds=1) == 1.0
    assert nocte.core.time.ms(seconds=1) == 1000.0
    assert nocte.core.time.ms(minutes=1) == 60_000.0
    assert nocte.core.time.ms(hours=1) == 3_600_000.0

    assert (
        nocte.core.time.ms(
            days=1,
            hours=2,
            minutes=3,
            seconds=4,
            milliseconds=5,
        )
        == 93_784_005.0
    )


@pytest.mark.parametrize(
    ('value', 'scale', 'expected'),
    [
        (1499.0, 'seconds', 1000.0),
        (1501.0, 'seconds', 2000.0),
        (61_000.0, 'minutes', 60_000.0),
        (91_000.0, 30_000.0, 90_000.0),
    ],
)
def test_ms_round(value, scale, expected):
    assert nocte.core.time.ms_round(value, scale) == expected


def test_ms_round_decimals():
    assert (
        nocte.core.time.ms_round(
            1250.0,
            scale='seconds',
            decimals=1,
        )
        == 1200.0
    )


@pytest.mark.parametrize(
    ('value', 'scale', 'floor', 'ceil'),
    [
        (61_000.0, 'minutes', 60_000.0, 120_000.0),
        (-61_000.0, 'minutes', -120_000.0, -60_000.0),
        (75.0, 50.0, 50.0, 100.0),
    ],
)
def test_ms_floor_ceil(value, scale, floor, ceil):
    assert nocte.core.time.ms_floor(value, scale) == floor
    assert nocte.core.time.ms_ceil(value, scale) == ceil


def test_ms_remainder():
    value = nocte.core.time.ms(days=8, hours=7)

    assert nocte.core.time.ms_remainder(
        value,
        scale='days',
    ) == nocte.core.time.ms(hours=7)


def test_ms_remainder_negative():
    assert (
        nocte.core.time.ms_remainder(
            -61_000,
            scale='minutes',
        )
        == 59_000.0
    )


@pytest.mark.parametrize(
    ('value', 'kwargs', 'expected'),
    [
        (0.0, {}, '00:00'),
        (1000.0, {}, '00:00:01'),
        (1001.0, {}, '00:00:01.001'),
        (-1500.0, {}, '-00:00:01.500'),
        (60_000.0, {'plus_sign': True}, '+00:01'),
        (
            nocte.core.time.ms(days=1, hours=2),
            {'show_days': True},
            '1d 02:00',
        ),
        (
            nocte.core.time.ms(days=1, hours=2),
            {'show_days': False},
            '26:00',
        ),
        (
            0.0,
            {'strip': False},
            '00:00:00.000',
        ),
    ],
)
def test_ms_to_str(value, kwargs, expected):
    assert nocte.core.time.ms_to_str(value, **kwargs) == expected


@pytest.mark.parametrize(
    ('timestamp', 'expected'),
    [
        ('00:00', 0.0),
        ('19:00', nocte.core.time.ms(hours=19)),
        ('19:00:15', nocte.core.time.ms(hours=19, seconds=15)),
        (
            '19:00:15.143',
            nocte.core.time.ms(hours=19, seconds=15, milliseconds=143),
        ),
        (
            '3d 19:00',
            nocte.core.time.ms(days=3, hours=19),
        ),
        ('+00:01', nocte.core.time.ms(minutes=1)),
        ('-00:10:30.500', -nocte.core.time.ms(minutes=10, seconds=30.5)),
    ],
)
def test_str_to_ms(timestamp, expected):
    assert nocte.core.time.str_to_ms(timestamp) == expected


@pytest.mark.parametrize(
    'timestamp',
    [
        '',
        'foo',
        '19',
        '19:00xxx',
        '12:60',
        '12:00:60',
        '3d',
    ],
)
def test_str_to_ms_rejects_invalid(timestamp):
    with pytest.raises(ValueError):
        nocte.core.time.str_to_ms(timestamp)


@pytest.mark.parametrize(
    'value',
    [
        0.0,
        1.0,
        -1.0,
        1001.0,
        -1001.0,
        nocte.core.time.ms(hours=19, minutes=4, seconds=3, milliseconds=27),
        nocte.core.time.ms(days=4, hours=7, milliseconds=123),
    ],
)
def test_ms_string_roundtrip(value):
    string = nocte.core.time.ms_to_str(
        value,
        strip=False,
        show_days=True,
    )

    assert nocte.core.time.str_to_ms(string) == value


@pytest.mark.parametrize(
    'scale',
    [
        0.0,
        -1.0,
        np.inf,
        np.nan,
    ],
)
def test_rounding_rejects_invalid_numeric_scale(scale):
    with pytest.raises(ValueError):
        nocte.core.time.ms_round(1000.0, scale=scale)
