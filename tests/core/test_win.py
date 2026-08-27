# tests/core/test_windows.py

import numpy as np
import pytest

import nocte.core.time
import nocte.core.windows

Win = nocte.core.windows.Win


def test_win():
    win = Win(-100, 500, ref=1000)

    assert win.start == -100.0
    assert win.stop == 500.0
    assert win.ref == 1000.0


def test_win_defaults_ref_to_zero():
    win = Win(-100, 500)

    assert win.ref == 0.0


def test_win_normalizes_to_float():
    win = Win(-1, 2, ref=3)

    assert isinstance(win.start, float)
    assert isinstance(win.stop, float)
    assert isinstance(win.ref, float)


@pytest.mark.parametrize(
    ('start', 'stop', 'ref'),
    [
        (np.nan, 1, 0),
        (0, np.nan, 0),
        (0, 1, np.nan),
        (-np.inf, 1, 0),
        (0, np.inf, 0),
    ],
)
def test_win_rejects_nonfinite_values(start, stop, ref):
    with pytest.raises(ValueError):
        Win(start, stop, ref=ref)


def test_win_rejects_negative_length():
    with pytest.raises(ValueError):
        Win(10, 5)


def test_empty_win_is_valid():
    win = Win(10, 10)

    assert win.is_empty()
    assert win.length == 0.0


def test_in_units():
    win = Win.in_units(-10, 20, 'minutes')

    assert win.start == nocte.core.time.ms(minutes=-10)
    assert win.stop == nocte.core.time.ms(minutes=20)
    assert win.ref == 0.0


def test_from_center():
    win = Win.from_center(
        center=1000,
        duration=400,
        ref=700,
    )

    assert win == Win(100, 500, ref=700)
    assert win.mid == 1000.0


def test_length():
    assert Win(-100, 500).length == 600.0


def test_time_at():
    win = Win(-100, 500, ref=1000)

    assert win.time_at('start') == 900.0
    assert win.time_at('ref') == 1000.0
    assert win.time_at('mid') == 1200.0
    assert win.time_at('stop') == 1500.0

    assert win.time_at(0.0) == 900.0
    assert win.time_at(0.25) == 1050.0
    assert win.time_at(1.0) == 1500.0


@pytest.mark.parametrize('q', [-0.1, 1.1])
def test_time_at_rejects_outside_window(q):
    with pytest.raises(ValueError):
        Win(0, 100).time_at(q)


def test_mid_is_in_enclosing_coordinate():
    win = Win(-100, 300, ref=1000)

    assert win.mid == 1100.0


def test_contains_is_half_open():
    win = Win(-100, 500, ref=1000)

    assert not win.contains(899)
    assert win.contains(900)
    assert win.contains(1499)
    assert not win.contains(1500)


def test_contains_array():
    win = Win(-100, 500, ref=1000)

    result = win.contains(np.array([899, 900, 1000, 1499, 1500]))

    np.testing.assert_array_equal(
        result,
        [False, True, True, True, False],
    )


def test_contained_in_with_different_refs():
    inner = Win(-50, 50, ref=1000)  # [950, 1050)
    outer = Win(-200, 200, ref=1000)  # [800, 1200)

    assert inner.contained_in(outer)
    assert not outer.contained_in(inner)


def test_contained_in_accounts_for_ref():
    inner = Win(0, 100, ref=1000)  # [1000, 1100)
    outer = Win(400, 600, ref=500)  # [900, 1100)

    assert inner.contained_in(outer)


def test_overlaps():
    assert Win(0, 100).overlaps(Win(50, 150))
    assert not Win(0, 100).overlaps(Win(100, 200))


def test_overlaps_accounts_for_ref():
    first = Win(0, 100, ref=1000)  # [1000, 1100)
    second = Win(0, 100, ref=1050)  # [1050, 1150)

    assert first.overlaps(second)


def test_before():
    win = Win(-100, 500, ref=1000)

    before = win.before(200)

    assert before == Win(-300, -100, ref=1000)


def test_before_with_offset():
    win = Win(0, 100, ref=1000)

    before = win.before(50, offset=20)

    assert before == Win(-70, -20, ref=1000)


def test_after():
    win = Win(-100, 500, ref=1000)

    after = win.after(200)

    assert after == Win(500, 700, ref=1000)


def test_after_with_offset():
    win = Win(0, 100, ref=1000)

    after = win.after(50, offset=20)

    assert after == Win(120, 170, ref=1000)


def test_centered():
    win = Win(-100, 300, ref=1000)

    centered = win.centered(100)

    assert centered == Win(50, 150, ref=1000)
    assert centered.mid == win.mid


def test_change_preserves_ref():
    win = Win(-100, 500, ref=1000)

    changed = win.change(pre=-50, post=100)

    assert changed == Win(-150, 600, ref=1000)


def test_expand():
    win = Win(-100, 500, ref=1000)

    assert win.expand(50) == Win(-150, 550, ref=1000)


def test_shrink():
    win = Win(-100, 500, ref=1000)

    assert win.shrink(50) == Win(-50, 450, ref=1000)


def test_shift_changes_ref_not_geometry():
    win = Win(-100, 500, ref=1000)

    shifted = win.shift(250)

    assert shifted == Win(-100, 500, ref=1250)
    assert shifted.length == win.length


def test_crop():
    win = Win(-100, 500, ref=1000)  # [900, 1500)
    other = Win(0, 300, ref=1000)  # [1000, 1300)

    cropped = win.crop(other)

    assert cropped == Win(0, 300, ref=1000)


def test_crop_preserves_self_ref():
    win = Win(-100, 500, ref=1000)  # [900, 1500)
    other = Win(0, 200, ref=1200)  # [1200, 1400)

    cropped = win.crop(other)

    assert cropped == Win(200, 400, ref=1000)


def test_crop_disjoint_returns_empty_window():
    win = Win(0, 100, ref=0)
    other = Win(200, 300, ref=0)

    cropped = win.crop(other)

    assert cropped == Win(200, 200, ref=0)
    assert cropped.is_empty()


def test_shift_to_fit_left():
    win = Win(-100, 100, ref=0)  # [-100, 100)
    outer = Win(0, 1000, ref=0)

    shifted = win.shift_to_fit(outer)

    assert shifted == Win(-100, 100, ref=100)


def test_shift_to_fit_right():
    win = Win(900, 1100, ref=0)
    outer = Win(0, 1000, ref=0)

    shifted = win.shift_to_fit(outer)

    assert shifted == Win(900, 1100, ref=-100)


def test_shift_to_fit_does_nothing_when_already_inside():
    win = Win(100, 200, ref=0)
    outer = Win(0, 1000, ref=0)

    assert win.shift_to_fit(outer) == win


def test_shift_to_fit_rejects_window_that_is_too_large():
    with pytest.raises(ValueError):
        Win(0, 200).shift_to_fit(Win(0, 100))


def test_take_centered():
    win = Win(-200, 400, ref=1000)

    result = win.take_centered(200)

    assert result.length == 200.0
    assert result.mid == win.mid


def test_take_centered_keeps_shorter_window_unchanged():
    win = Win(-100, 100, ref=1000)

    assert win.take_centered(500) == win


def test_arange_is_half_open():
    win = Win(-100, 200, ref=1000)

    result = win.arange(100)

    np.testing.assert_array_equal(
        result,
        [900, 1000, 1100],
    )


@pytest.mark.parametrize('step', [0, -1])
def test_arange_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        Win(0, 100).arange(step)


def test_round():
    win = Win(-61_234, 61_234, ref=123)

    rounded = win.round(scale='minutes')

    assert rounded == Win(-60_000, 60_000, ref=123)


def test_round_loose():
    win = Win(-61_000, 61_000, ref=123)

    rounded = win.round_loose('minutes')

    assert rounded == Win(-120_000, 120_000, ref=123)


def test_round_tight():
    win = Win(-61_000, 61_000, ref=123)

    rounded = win.round_tight('minutes')

    assert rounded == Win(-60_000, 60_000, ref=123)


def test_empty_win_contains_no_time():
    win = Win(5, 5)

    assert not win.contains(5)


@pytest.mark.parametrize(
    'empty',
    [
        Win(0, 0),
        Win(5, 5),
        Win(10, 10),
        Win(0, 0, ref=5),
    ],
)
def test_empty_win_does_not_overlap(empty):
    win = Win(0, 10)

    assert not empty.overlaps(win)
    assert not win.overlaps(empty)
