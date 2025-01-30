import numpy as np
import pandas as pd

from nocte.timeslice import Windows, _classify_events_exclusive


def test_build_around():
    Windows.build_around([0, 1, 2], (-1, 10))
    Windows.build_around(np.array([0, 1, 2]), (-1, 10))
    Windows.build_around(pd.Series([0, 1, 2]), (-1, 10))


def test_build_around_multiple():
    Windows.build_around_multiple([0, 1, 2], pre=(-10, 0), post=(0, 10))


# noinspection PyTypeChecker
def test_build_between():
    wins = Windows.build_between([1, 2, 3])
    assert len(wins) == 2

    wins = Windows.build_between([1, 2, 3], start=0)
    assert len(wins) == 3

    wins = Windows.build_between([1, 2, 3], start=0, stop=4)
    assert len(wins) == 4
    assert len(wins.wins.columns) == 3

    wins = Windows.build_between(pd.Series([1, 2, 3]), start=0, stop=4)
    assert len(wins) == 4
    assert len(wins.wins.columns) == 3
    print(wins)


def test_build_between_df():
    df = pd.DataFrame({'time': [1, 2, 3], 'else': ['a', 'b', 'c']})
    # noinspection PyTypeChecker
    wins = Windows.build_between(df, start=0, stop=4)
    assert len(wins) == 4
    assert np.all(wins.wins.columns == ['start', 'stop', 'ref', 'start_else', 'stop_else'])


def test_build_from_contiguous_values():
    wins = Windows.build_from_contiguous_values([0, 0, 0, 1, 1, 1, 0, 2, 2, 2], include_right=False)
    assert (wins.total_by_cat() == pd.Series({0: 4, 1: 3, 2: 3})).all()


def test_classify_exclusive_ref_cols():
    kwargs = dict(
        merge_wincols=None,
        right=False,
    )
    wins = Windows.build_around(
        pd.Series([100, 200, 300], index=['z', 'a', 'k']), (-50, +50)
    )
    wins = wins.wins

    assert _classify_events_exclusive(wins, [], ref_col=['start', 'stop'], **kwargs).shape == (0, 3)


def test_classify_exclusive():
    kwargs = dict(
        ref_col='ref',
        merge_wincols=None,
        right=False,
    )
    wins = Windows.build_around(
        pd.Series([100, 200, 300], index=['z', 'a', 'k']), (-50, +50)
    )
    wins = wins.wins

    assert _classify_events_exclusive(wins, [], **kwargs).shape == (0, 2)

    # outside on the left and right
    assert _classify_events_exclusive(wins, [0, 400], **kwargs).shape == (0, 2)

    # valid classifications
    assert np.all(_classify_events_exclusive(wins, [0, 75, 400], **kwargs)[['win_idx', 'delay']] == ('z', -25.))
    assert np.all(_classify_events_exclusive(wins, [0, 200, 400], **kwargs)[['win_idx', 'delay']] == ('a', 0.))
    assert np.all(_classify_events_exclusive(wins, [0, 325, 400], **kwargs)[['win_idx', 'delay']] == ('k', +25.))

    # respect event indices
    delays = _classify_events_exclusive(wins, pd.Series(dict(x=75, y=200, z=325)), **kwargs)
    assert delays.index.is_unique
    assert np.all(delays['win_idx'] == np.array(['z', 'a', 'k']))
    assert np.all(delays.index == np.array(['x', 'y', 'z']))


def test_classify_nonexclusive():
    # check exclusive works
    wins = Windows.build_around(pd.Series(dict(a=100, b=200, c=300)), (-50, +50))
    delays = wins.classify_events(pd.Series(dict(x=150, y=250)))
    assert delays.index.is_unique
    delays = delays.sort_values('win_idx').sort_index()
    assert np.all(delays.index == ['x', 'y'])
    assert np.all(delays['win_idx'].values == ['b', 'c'])  # left-inclusive

    # check exclusive works
    wins = Windows.build_around(pd.Series(dict(a=100, b=200, c=300)), (-50, +50))
    delays = wins.classify_events(pd.Series(dict(x=150, y=250)), right=True)
    assert delays.index.is_unique
    delays = delays.sort_values('win_idx').sort_index()
    assert np.all(delays.index == ['x', 'y'])
    assert np.all(delays['win_idx'].values == ['a', 'b'])  # left-inclusive

    # check non-exclusive works
    wins = Windows.build_around(pd.Series(dict(a=100, b=200, c=300)), (-100, +100))
    delays = wins.classify_events(pd.Series(dict(x=150, y=250)))
    assert not delays.index.is_unique
    delays = delays.sort_values('win_idx').sort_index()
    assert np.all(delays.index == ['x', 'x', 'y', 'y'])
    assert np.all(delays['win_idx'].values == ['a', 'b', 'b', 'c'])
