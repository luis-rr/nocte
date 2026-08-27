import numpy as np
import pandas as pd
import pytest

import nocte.core.collections


class _DummyCollection(nocte.core.collections.Collection):
    def __init__(
        self,
        data: np.ndarray,
        meta: pd.DataFrame,
    ):
        self.data = np.asarray(data)
        self.meta = meta.copy()

        self._validate_meta()

    def __len__(self) -> int:
        return len(self.data)

    def _take_pos(
        self,
        positions: np.ndarray,
    ):
        return type(self)(
            data=self.data[positions],
            meta=self.meta.iloc[positions].copy(),
        )


@pytest.fixture
def collection():
    meta = pd.DataFrame(
        {
            'condition': ['a', 'b', np.nan, 'a'],
            'group': ['x', 'x', 'y', 'y'],
            'value': [3.0, 1.0, 2.0, 4.0],
        },
        index=pd.Index(
            [10, 30, 20, 40],
            name='item_id',
        ),
    )

    return _DummyCollection(
        data=np.array([100, 300, 200, 400]),
        meta=meta,
    )


def assert_aligned(collection):
    np.testing.assert_array_equal(
        collection.data,
        collection.index.to_numpy() * 10,
    )


def test_validate_meta_length():
    meta = pd.DataFrame(
        index=pd.Index([10, 20]),
    )

    with pytest.raises(ValueError, match='meta has 2 rows'):
        _DummyCollection(
            data=np.array([100]),
            meta=meta,
        )


def test_validate_meta_unique_index():
    meta = pd.DataFrame(
        index=pd.Index([10, 10]),
    )

    with pytest.raises(ValueError, match='index must be unique'):
        _DummyCollection(
            data=np.array([100, 100]),
            meta=meta,
        )


def test_metadata_access(collection):
    assert collection.columns.equals(collection.meta.columns)
    assert collection['value'].equals(collection.meta['value'])

    collection['new'] = [1, 2, 3, 4]

    assert collection.meta['new'].tolist() == [1, 2, 3, 4]


def test_sel_index_preserves_requested_order(collection):
    selected = collection.sel([40, 10])

    assert selected.index.tolist() == [40, 10]
    assert_aligned(selected)


def test_sel_index_invert_preserves_original_order(collection):
    selected = collection.sel([30, 40], invert=True)

    assert selected.index.tolist() == [10, 20]
    assert_aligned(selected)


def test_sel_index_missing(collection):
    with pytest.raises(KeyError, match='item labels not found'):
        collection.sel([10, 999])


def test_sel_match(collection):
    selected = collection.sel(condition='a')

    assert selected.index.tolist() == [10, 40]
    assert_aligned(selected)


def test_sel_match_nan(collection):
    selected = collection.sel(condition=np.nan)

    assert selected.index.tolist() == [20]
    assert_aligned(selected)


def test_sel_match_requires_scalar(collection):
    with pytest.raises(TypeError, match='use sel_in'):
        collection.sel_match(
            condition=['a', 'b'],
        )


def test_sel_in(collection):
    selected = collection.sel_in(
        condition=['a', 'b'],
    )

    assert selected.index.tolist() == [10, 30, 40]
    assert_aligned(selected)


def test_sel_between(collection):
    selected = collection.sel_between(
        value=(1.5, 3.5),
    )

    assert selected.index.tolist() == [10, 20]
    assert_aligned(selected)


def test_selection_any_and_invert(collection):
    selected = collection.sel_match(
        condition='a',
        group='x',
        how='any',
        invert=True,
    )

    assert selected.index.tolist() == [20]
    assert_aligned(selected)


def test_sel_mask_series(collection):
    mask = collection['value'] >= 2

    selected = collection.sel_mask(mask)

    assert selected.index.tolist() == [10, 20, 40]
    assert_aligned(selected)


@pytest.mark.parametrize(
    ('mask', 'error'),
    [
        ([1, 0, 1, 0], TypeError),
        ([True, False], ValueError),
        (
            np.array(
                [
                    [True],
                    [False],
                    [True],
                    [False],
                ]
            ),
            ValueError,
        ),
    ],
)
def test_sel_mask_rejects_invalid_masks(
    collection,
    mask,
    error,
):
    with pytest.raises(error):
        collection.sel_mask(mask)


def test_sel_mask_requires_matching_series_index(
    collection,
):
    mask = pd.Series(
        [True, False, True, False],
        index=[40, 20, 30, 10],
    )

    with pytest.raises(
        ValueError,
        match='mask index does not match',
    ):
        collection.sel_mask(mask)


def test_sort_values_reorders_meta_and_payload(collection):
    sorted_collection = collection.sort_values(
        'value',
    )

    assert sorted_collection.index.tolist() == [
        30,
        20,
        10,
        40,
    ]
    assert sorted_collection['value'].tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert_aligned(sorted_collection)


def test_sort_index_reorders_meta_and_payload(collection):
    sorted_collection = collection.sort_index()

    assert sorted_collection.index.tolist() == [
        10,
        20,
        30,
        40,
    ]
    assert_aligned(sorted_collection)


def test_sort_allows_explicit_non_inplace(collection):
    selected = collection.sort_values(
        'value',
        inplace=False,
        ignore_index=False,
    )

    assert_aligned(selected)


def test_sample_is_deterministic_with_random_state(
    collection,
):
    first = collection.sample(
        n=2,
        random_state=123,
        replace=False,
    )
    second = collection.sample(
        n=2,
        random_state=123,
        replace=False,
    )

    assert first.index.equals(second.index)
    assert_aligned(first)


def test_shuffle_preserves_items(collection):
    shuffled = collection.shuffle()

    assert sorted(shuffled.index) == sorted(collection.index)
    assert_aligned(shuffled)


@pytest.mark.parametrize(
    ('method', 'kwargs'),
    [
        ('sort_values', {'inplace': True}),
        ('sort_values', {'ignore_index': True}),
        ('sort_index', {'inplace': True}),
        ('sort_index', {'ignore_index': True}),
        ('sample', {'ignore_index': True}),
        ('sample', {'replace': True}),
    ],
)
def test_operations_reject_broken_index_semantics(
    collection,
    method,
    kwargs,
):
    func = getattr(collection, method)

    args = ('value',) if method == 'sort_values' else ()

    with pytest.raises(TypeError):
        func(*args, **kwargs)
