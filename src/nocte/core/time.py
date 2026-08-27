import datetime
import re
import typing

import numpy as np

S_TO_MS = 1e3
MS_TO_S = 1.0 / S_TO_MS

TimeScale = typing.Literal[
    'microseconds',
    'milliseconds',
    'seconds',
    'minutes',
    'hours',
    'days',
    'weeks',
]

_MS_PER_SCALE: dict[TimeScale, float] = {
    'microseconds': 1e-3,
    'milliseconds': 1.0,
    'seconds': 1e3,
    'minutes': 60e3,
    'hours': 60 * 60e3,
    'days': 24 * 60 * 60e3,
    'weeks': 7 * 24 * 60 * 60e3,
}


def to_ms(t: float | datetime.timedelta) -> float:
    """Convert a numeric millisecond value or timedelta to float milliseconds."""
    if isinstance(t, datetime.timedelta):
        return t.total_seconds() * S_TO_MS

    return float(t)


def ms(
    *,
    weeks: float = 0.0,
    days: float = 0.0,
    hours: float = 0.0,
    minutes: float = 0.0,
    seconds: float = 0.0,
    milliseconds: float = 0.0,
    microseconds: float = 0.0,
) -> float:
    """Build a millisecond duration using human-readable time units."""
    return float(
        weeks * 604_800_000
        + days * 86_400_000
        + hours * 3_600_000
        + minutes * 60_000
        + seconds * 1_000
        + milliseconds
        + microseconds / 1_000
    )


def scale_to_ms(scale: TimeScale | float) -> float:
    """Resolve a named or numeric time scale to milliseconds."""
    if isinstance(scale, str):
        scale_ms = _MS_PER_SCALE[scale]
    else:
        scale_ms = float(scale)

    if not np.isfinite(scale_ms) or scale_ms <= 0:
        raise ValueError('Time scale must be finite and positive')

    return scale_ms


def ms_round(
    value: float,
    scale: TimeScale | float = 'milliseconds',
    decimals: int = 0,
) -> float:
    """Round a millisecond value to a given time scale."""
    scale_ms = scale_to_ms(scale)
    return float(np.round(value / scale_ms, decimals=decimals) * scale_ms)


def ms_floor(
    value: float,
    scale: TimeScale | float = 'milliseconds',
) -> float:
    """Round a millisecond value down to a given time scale."""
    scale_ms = scale_to_ms(scale)
    return float(np.floor(value / scale_ms) * scale_ms)


def ms_ceil(
    value: float,
    scale: TimeScale | float = 'milliseconds',
) -> float:
    """Round a millisecond value up to a given time scale."""
    scale_ms = scale_to_ms(scale)
    return float(np.ceil(value / scale_ms) * scale_ms)


def ms_remainder(
    value: float,
    scale: TimeScale | float = 'days',
) -> float:
    """Return the remainder after removing full multiples of a time scale."""
    return float(value - ms_floor(value, scale=scale))


def ms_to_str(
    value: float,
    plus_sign: bool = False,
    strip: bool = True,
    show_days: bool = False,
) -> str:
    """
    Format milliseconds as `[+/-][DDd ]HH:MM[:SS[.sss]]`.

    Formatting has millisecond precision.
    """
    value = to_ms(value)

    sign = ''
    if value < 0:
        sign = '-'
    elif plus_sign:
        sign = '+'

    total_ms = int(np.round(abs(value)))

    if show_days:
        days, total_ms = divmod(total_ms, int(ms(days=1)))
    else:
        days = None

    hours, total_ms = divmod(total_ms, int(ms(hours=1)))
    minutes, total_ms = divmod(total_ms, int(ms(minutes=1)))
    seconds, milliseconds = divmod(total_ms, int(ms(seconds=1)))

    desc = sign

    if days is not None:
        desc += f'{days}d '

    desc += f'{hours:02d}:{minutes:02d}'

    if seconds > 0 or milliseconds > 0 or not strip:
        desc += f':{seconds:02d}'

        if milliseconds > 0 or not strip:
            desc += f'.{milliseconds:03d}'

    return desc


_TIMESTAMP_PATTERN = re.compile(
    r'(?P<sign>[+-])?'
    r'(?:(?P<days>\d+)d\s*)?'
    r'(?P<hours>\d+):'
    r'(?P<minutes>\d{2})'
    r'(?::(?P<seconds>\d{2})'
    r'(?:\.(?P<milliseconds>\d{1,3}))?'
    r')?'
)


def str_to_ms(timestamp: str) -> float:
    """
    Parse `[+/-][DDd ]HH:MM[:SS[.sss]]` into milliseconds.

    Examples:
        '19:00'
        '19:00:15'
        '19:00:15.143'
        '3d 19:00'
        '-00:10:30.500'
    """
    match = _TIMESTAMP_PATTERN.fullmatch(timestamp.strip())

    if match is None:
        raise ValueError(f'Invalid timestamp format: {timestamp!r}')

    minutes = int(match.group('minutes'))
    seconds = int(match.group('seconds') or 0)

    if minutes >= 60:
        raise ValueError(f'Invalid minutes in timestamp: {timestamp!r}')

    if seconds >= 60:
        raise ValueError(f'Invalid seconds in timestamp: {timestamp!r}')

    milliseconds_str = match.group('milliseconds') or ''
    milliseconds = int(milliseconds_str.ljust(3, '0')) if milliseconds_str else 0

    value = ms(
        days=int(match.group('days') or 0),
        hours=int(match.group('hours')),
        minutes=minutes,
        seconds=seconds,
        milliseconds=milliseconds,
    )

    if match.group('sign') == '-':
        value = -value

    return value
