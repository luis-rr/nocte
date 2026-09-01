import typing

import numpy as np
import pandas as pd
import pytest

from nocte._core.collection import Collection
from nocte._core.matching import Matches


class _TestCollection(Collection[int]):
    """Minimal metadata-only Collection for matching tests."""

    def __init__(self, meta: pd.DataFrame):
        self.meta = meta.copy()
        self._validate_meta(len(self.meta))

    def _sel_pos(self, positions: np.ndarray) -> typing.Self:
        return self.__class__(
            self.meta.iloc[positions],
        )

    def _get_pos(self, position: int) -> int:
        return int(self.index.to_numpy(copy=False)[position])


def _collection(
    ids,
    *,
    name: str,
    **meta,
) -> _TestCollection:
    return _TestCollection(
        pd.DataFrame(
            meta,
            index=pd.Index(
                ids,
                dtype=np.int64,
                name=name,
            ),
        )
    )


def test_from_meta_many_to_many():
    left = _collection(
        [10, 11, 12],
        name='trace',
        exp=['a', 'a', 'b'],
    )
    right = _collection(
        [20, 21, 22, 23],
        name='win',
        exp=['a', 'b', 'a', 'c'],
    )

    left_meta = left.meta.copy()
    right_meta = right.meta.copy()

    matches = Matches.from_meta(
        left,
        right,
        by='exp',
    )

    expected = pd.DataFrame(
        {
            'trace': [10, 10, 11, 11, 12],
            'win': [20, 22, 20, 22, 21],
        },
        index=pd.RangeIndex(
            5,
            name='match',
        ),
    )

    pd.testing.assert_frame_equal(
        matches.to_frame(),
        expected,
    )

    assert matches.left_name == 'trace'
    assert matches.right_name == 'win'
    assert matches.by == ('exp',)
    assert matches.name == 'match'

    pd.testing.assert_frame_equal(
        left.meta,
        left_meta,
    )
    pd.testing.assert_frame_equal(
        right.meta,
        right_meta,
    )


def test_from_meta_no_matches():
    left = _collection(
        [10, 11],
        name='trace',
        exp=['a', 'b'],
    )
    right = _collection(
        [20, 21],
        name='win',
        exp=['c', 'd'],
    )

    matches = Matches.from_meta(
        left,
        right,
        by='exp',
    )

    assert matches.empty
    assert len(matches.meta) == 0
    assert matches.name == 'match'
    assert matches.left_name == 'trace'
    assert matches.right_name == 'win'
    assert matches.by == ('exp',)
    assert matches.left.empty
    assert matches.right.empty


def test_from_product():
    left = _collection(
        [4, 8],
        name='trace',
    )
    right = _collection(
        [10, 20, 30],
        name='win',
    )

    matches = Matches.from_product(
        left,
        right,
    )

    expected = pd.DataFrame(
        {
            'trace': [4, 4, 4, 8, 8, 8],
            'win': [10, 20, 30, 10, 20, 30],
        },
        index=pd.RangeIndex(
            6,
            name='match',
        ),
    )

    pd.testing.assert_frame_equal(
        matches.to_frame(),
        expected,
    )


def test_from_identity():
    collection = _collection(
        [4, 8, 12],
        name='beta',
    )

    matches = Matches.from_identity(
        collection,
    )

    expected = pd.DataFrame(
        {
            'left_beta': [4, 8, 12],
            'right_beta': [4, 8, 12],
        },
        index=pd.RangeIndex(
            3,
            name='match',
        ),
    )

    pd.testing.assert_frame_equal(
        matches.to_frame(),
        expected,
    )


def test_from_combinations():
    collection = _collection(
        [4, 8, 12],
        name='beta',
    )

    matches = Matches.from_combinations(
        collection,
    )

    expected = pd.DataFrame(
        {
            'left_beta': [4, 4, 8],
            'right_beta': [8, 12, 12],
        },
        index=pd.RangeIndex(
            3,
            name='match',
        ),
    )

    pd.testing.assert_frame_equal(
        matches.to_frame(),
        expected,
    )


def test_selection_preserves_relation_state_and_alignment():
    left = _collection(
        [10, 11, 12],
        name='trace',
        exp=['a', 'a', 'b'],
    )
    right = _collection(
        [20, 21, 22],
        name='win',
        exp=['a', 'b', 'a'],
    )

    matches = Matches.from_meta(
        left,
        right,
        by='exp',
    )
    matches.meta['quality'] = np.arange(len(matches))

    selected = matches.sel_index(
        [2, 0],
    )

    assert selected.left_name == 'trace'
    assert selected.right_name == 'win'
    assert selected.by == ('exp',)

    pd.testing.assert_index_equal(
        selected.index,
        pd.Index(
            [2, 0],
            name='match',
        ),
    )

    assert selected.get(2) == matches.get(2)
    assert selected.get(0) == matches.get(0)

    pd.testing.assert_series_equal(
        selected.meta['quality'],
        matches.meta.loc[[2, 0], 'quality'],
    )


def test_rename_changes_only_match_identity_namespace():
    left = _collection(
        [1, 2],
        name='beta',
    )
    right = _collection(
        [10, 20],
        name='win',
    )

    matches = Matches.from_product(
        left,
        right,
    )

    renamed = matches.rename(
        'xcorr',
    )

    assert renamed.name == 'xcorr'
    assert renamed.left_name == 'beta'
    assert renamed.right_name == 'win'
    assert renamed.by is None

    pd.testing.assert_index_equal(
        renamed.index,
        matches.index.rename('xcorr'),
    )

    np.testing.assert_array_equal(
        renamed.left.to_numpy(),
        matches.left.to_numpy(),
    )
    np.testing.assert_array_equal(
        renamed.right.to_numpy(),
        matches.right.to_numpy(),
    )


def test_source_ids_resolve_to_repeated_positions():
    left = _collection(
        [10, 11, 12],
        name='trace',
        exp=['a', 'a', 'b'],
    )
    right = _collection(
        [20, 21, 22],
        name='win',
        exp=['a', 'b', 'a'],
    )

    matches = Matches.from_meta(
        left,
        right,
        by='exp',
    )

    np.testing.assert_array_equal(
        matches.left_positions(left),
        [0, 0, 1, 1, 2],
    )
    np.testing.assert_array_equal(
        matches.right_positions(right),
        [0, 2, 0, 2, 1],
    )


def test_empty_collections_are_valid():
    empty_left = _collection(
        [],
        name='trace',
        exp=pd.Series(dtype='object'),
    )
    empty_right = _collection(
        [],
        name='win',
        exp=pd.Series(dtype='object'),
    )
    right = _collection(
        [20, 21],
        name='win',
        exp=['a', 'b'],
    )

    relations = [
        Matches.from_meta(
            empty_left,
            empty_right,
            by='exp',
        ),
        Matches.from_product(
            empty_left,
            right,
        ),
        Matches.from_product(
            right,
            empty_left,
        ),
        Matches.from_identity(
            empty_left,
        ),
        Matches.from_combinations(
            empty_left,
        ),
    ]

    for matches in relations:
        assert matches.empty
        assert len(matches) == 0
        assert len(matches.meta) == 0
        assert matches.index.is_unique


def test_from_meta_requires_matching_columns():
    left = _collection(
        [1],
        name='trace',
        exp=['a'],
    )
    right = _collection(
        [2],
        name='win',
        animal=['a'],
    )

    with pytest.raises(
        KeyError,
        match='right collection is missing',
    ):
        Matches.from_meta(
            left,
            right,
            by='exp',
        )


def test_from_meta_rejects_identity_name_as_matching_column():
    left = _collection(
        [1],
        name='exp',
        exp=['a'],
    )
    right = _collection(
        [2],
        name='win',
        exp=['a'],
    )

    with pytest.raises(
        ValueError,
        match='identity name',
    ):
        Matches.from_meta(
            left,
            right,
            by='exp',
        )
