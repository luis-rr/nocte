import numpy as np
import pandas as pd
import pytest

from nocte.core.traces import Traces


def test_lookup_each_aligns_series_by_trace_identity():
    traces = Traces.from_array(
        np.array([[0, 1, 2], [10, 11, 12]]),
        hz=1_000,
        meta=pd.DataFrame(index=pd.Index([10, 20], name='trace_id')),
    )

    values = traces.lookup_each(pd.Series([2, 1], index=[20, 10]))

    np.testing.assert_array_equal(values, [1, 12])


@pytest.mark.parametrize('index', [[10], [10, 20, 30], [10, 10]])
def test_lookup_each_rejects_nonmatching_trace_identities(index):
    traces = Traces.from_array(
        np.array([[0, 1], [10, 11]]),
        hz=1_000,
        meta=pd.DataFrame(index=pd.Index([10, 20], name='trace_id')),
    )

    with pytest.raises(ValueError, match='exactly the collection index'):
        traces.lookup_each(pd.Series([1] * len(index), index=index))


def test_copy_shares_immutable_payload():
    traces = Traces.from_array(np.array([[1, 2, 3]]), hz=1_000)

    copied = traces.copy()

    assert copied._data is traces._data
    assert copied.meta is not traces.meta
    np.testing.assert_array_equal(copied.values, traces.values)


def test_shift_shares_immutable_payload_and_changes_coordinate():
    traces = Traces.from_array(np.array([[1, 2, 3]]), hz=1_000, start=10)

    shifted = traces.shift(5)

    assert shifted._data is not traces._data
    assert shifted._data.values is traces._data.values
    assert shifted.start == 15
    assert traces.start == 10
    np.testing.assert_array_equal(shifted.values, traces.values)
