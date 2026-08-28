import numpy as np
import pandas as pd
import pytest

from nocte.core.windows import Win

# -----------------------------------------------------------------------------
# construction and geometry


def test_win_normalizes_to_float_and_defaults_ref():
    win = Win(1, 3)

    assert win == Win(1.0, 3.0, ref=0.0)
    assert isinstance(win.start, float)
    assert isinstance(win.stop, float)
    assert isinstance(win.ref, float)


@pytest.mark.parametrize(
    ('start', 'stop', 'ref'),
    [
        (np.nan, 1, 0),
        (0, np.inf, 0),
        (0, 1, -np.inf),
    ],
)
def test_win_rejects_nonfinite_geometry(start, stop, ref):
    with pytest.raises(ValueError):
        Win(start, stop, ref=ref)


def test_win_rejects_negative_length():
    with pytest.raises(ValueError):
        Win(2, 1)


def test_empty_win_is_valid():
    win = Win(5, 5, ref=10)

    assert win.is_empty()
    assert win.length == 0


def test_in_units():
    assert Win.in_units(-1, 2, 'seconds') == Win(-1000, 2000)


def test_from_center():
    assert Win.from_center(100, 20, ref=50) == Win(40, 60, ref=50)


def test_from_center_rejects_negative_duration():
    with pytest.raises(ValueError):
        Win.from_center(0, -1)


def test_geometry_is_expressed_in_enclosing_coordinate():
    win = Win(-10, 30, ref=100)

    assert win.length == 40
    assert win.mid == 110
    assert win.time_at('start') == 90
    assert win.time_at('ref') == 100
    assert win.time_at('mid') == 110
    assert win.time_at('stop') == 130
    assert win.time_at(0.25) == 100


@pytest.mark.parametrize('q', [-0.01, 1.01, 'unknown'])
def test_time_at_rejects_invalid_position(q):
    with pytest.raises(ValueError):
        Win(0, 10).time_at(q)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# evaluation


def test_contains_is_half_open():
    win = Win(0, 10, ref=100)

    assert win.contains(100)
    assert win.contains(109.999)
    assert not win.contains(110)
    assert 100 in win
    assert 110 not in win


def test_empty_win_contains_no_time():
    win = Win(5, 5)

    assert not win.contains(5)


def test_contains_many_preserves_input_index():
    times = pd.Series(
        [99, 100, 109, 110],
        index=pd.Index([10, 20, 30, 40], name='event_id'),
    )

    result = Win(0, 10, ref=100).contains_many(times)

    pd.testing.assert_series_equal(
        result,
        pd.Series(
            [False, True, True, False],
            index=times.index,
            name='contains',
        ),
    )


def test_contains_many_array_gets_default_index():
    result = Win(0, 10).contains_many([0, 10])

    pd.testing.assert_series_equal(
        result,
        pd.Series([True, False], name='contains'),
    )


def test_contained_in_accounts_for_reference():
    inner = Win(-5, 5, ref=100)
    outer = Win(90, 110)

    assert inner.contained_in(outer)
    assert not outer.contained_in(inner)


def test_overlaps_is_half_open_and_accounts_for_reference():
    win = Win(0, 10, ref=100)

    assert win.overlaps(Win(105, 115))
    assert not win.overlaps(Win(110, 120))
    assert not Win(105, 105).overlaps(win)


# -----------------------------------------------------------------------------
# relative construction


def test_around_uses_selected_point_and_template_reference():
    base = Win(-10, 30, ref=100)

    result = base.around(Win(-2, 4, ref=3), q='ref')

    assert result == Win(-2, 4, ref=103)


def test_centered_can_target_reference():
    result = Win(-10, 30, ref=100).centered(20, q='ref')

    assert result == Win(-10, 10, ref=100)


def test_before_and_after_support_offset_and_selected_point():
    win = Win(0, 20, ref=100)

    before = win.before(10, offset=2, q='mid')
    after = win.after(10, offset=2, q='mid')

    assert before == Win(-12, -2, ref=110)
    assert after == Win(2, 12, ref=110)


@pytest.mark.parametrize('method', ['before', 'after'])
def test_relative_construction_rejects_negative_duration(method):
    with pytest.raises(ValueError):
        getattr(Win(0, 10), method)(-1)


# -----------------------------------------------------------------------------
# geometry transformations


def test_change_preserves_reference():
    win = Win(-10, 20, ref=100)

    assert win.change(pre=-5, post=10) == Win(-15, 30, ref=100)


def test_shrink_and_expand():
    win = Win(0, 20, ref=100)

    assert win.shrink(5) == Win(5, 15, ref=100)
    assert win.expand(5) == Win(-5, 25, ref=100)


@pytest.mark.parametrize('method', ['shrink', 'expand'])
def test_resize_rejects_negative_duration(method):
    with pytest.raises(ValueError):
        getattr(Win(0, 10), method)(-1)


def test_shift_moves_interval_but_reanchor_preserves_it():
    win = Win(-10, 20, ref=100)

    shifted = win.shift(50)
    reanchored = win.reanchor('start')

    assert shifted == Win(-10, 20, ref=150)
    assert reanchored == Win(0, 30, ref=90)
    assert reanchored.time_at('start') == win.time_at('start')
    assert reanchored.time_at('stop') == win.time_at('stop')


def test_crop_preserves_reference_and_returns_intersection():
    win = Win(-10, 20, ref=100)

    assert win.crop(Win(95, 105)) == Win(-5, 5, ref=100)


def test_crop_disjoint_returns_empty_at_nearest_boundary():
    win = Win(0, 10)

    assert win.crop(Win(20, 30)) == Win(20, 20)
    assert win.crop(Win(-30, -20)) == Win(-20, -20)


def test_shift_within_moves_by_minimum_required_amount():
    bounds = Win(0, 100)

    assert Win(-20, 30).shift_within(bounds) == Win(-20, 30, ref=20)
    assert Win(80, 130).shift_within(bounds) == Win(80, 130, ref=-30)

    inside = Win(20, 40)
    assert inside.shift_within(bounds) is inside


def test_shift_within_rejects_window_that_is_too_long():
    with pytest.raises(ValueError):
        Win(0, 20).shift_within(Win(0, 10))


@pytest.mark.parametrize(
    ('q', 'expected'),
    [
        ('start', Win(0, 40)),
        ('mid', Win(30, 70)),
        ('stop', Win(60, 100)),
        (0.25, Win(15, 55)),
    ],
)
def test_cap_preserves_selected_position(q, expected):
    assert Win(0, 100).cap(40, q=q) == expected


def test_cap_leaves_shorter_window_unchanged():
    win = Win(0, 20)

    assert win.cap(40) is win


@pytest.mark.parametrize(
    ('max_duration', 'q'),
    [
        (-1, 'mid'),
        (np.inf, 'mid'),
        (10, -0.1),
        (10, 1.1),
        (10, 'ref'),
    ],
)
def test_cap_rejects_invalid_arguments(max_duration, q):
    with pytest.raises(ValueError):
        Win(0, 100).cap(max_duration, q=q)  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# time generation and quantization


def test_arange_is_half_open():
    np.testing.assert_array_equal(
        Win(0, 10, ref=100).arange(3),
        [100, 103, 106, 109],
    )


@pytest.mark.parametrize('step', [0, -1, np.inf, np.nan])
def test_arange_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        Win(0, 10).arange(step)


def test_round_can_target_individual_edges():
    win = Win(1234, 2678)

    assert win.round(scale='seconds') == Win(1000, 3000)
    assert win.round(start=True, stop=False, scale='seconds') == Win(1000, 2678)


def test_floor_and_ceil():
    win = Win(1234, 2678)

    assert win.floor(scale='seconds') == Win(1000, 2000)
    assert win.ceil(scale='seconds') == Win(2000, 3000)


def test_snap_modes():
    win = Win(1200, 2400)

    assert win.snap(scale='seconds') == Win(1000, 2000)
    assert win.snap('loose', scale='seconds') == Win(1000, 3000)
    assert win.snap('tight', scale='seconds') == Win(2000, 2000)


def test_snap_rejects_unknown_mode():
    with pytest.raises(ValueError):
        Win(0, 10).snap('unknown')  # type: ignore[arg-type]
