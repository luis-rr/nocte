import numpy as np
import pandas as pd
import pytest

from nocte._core.collection import Collection


class _DummyCollection(Collection[float]):
    def __init__(self, data: np.ndarray, meta: pd.DataFrame):
        self.data = np.asarray(data)
        self.meta = meta.copy()
        self._validate_meta(len(self.data))

    def _sel_pos(self, positions: np.ndarray):
        return type(self)(self.data[positions], self.meta.iloc[positions])

    def _get_pos(self, position: int) -> float:
        return self.data[position]


@pytest.fixture
def collection():
    return _DummyCollection(
        np.array([100, 200]),
        pd.DataFrame(index=pd.Index([10, 20], name='item')),
    )


def assert_aligned(collection):
    np.testing.assert_array_equal(
        collection.data,
        collection.index.to_numpy() * 10,
    )


def test_name_is_the_metadata_index_name(collection):
    assert collection.name == 'item'


@pytest.mark.parametrize('name', [None, '', 1])
def test_collection_requires_a_non_empty_string_index_name(name):
    with pytest.raises(ValueError, match='non-empty string'):
        _DummyCollection(
            np.array([100]),
            pd.DataFrame(index=pd.Index([10], name=name)),
        )


def test_rename_preserves_item_identities_and_payload(collection):
    renamed = collection.rename('observation')

    assert renamed.name == 'observation'
    assert renamed.index.tolist() == collection.index.tolist()
    assert_aligned(renamed)
    assert collection.name == 'item'


@pytest.mark.parametrize('name', ['', None, 1])
def test_rename_requires_a_non_empty_string(collection, name):
    with pytest.raises(ValueError, match='non-empty string'):
        collection.rename(name)
