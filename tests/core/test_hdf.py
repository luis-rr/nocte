import pathlib

import h5py
import numpy as np
import pandas as pd
import pytest

import nocte.core.hdf
import nocte.core.traces
import nocte.core.windows

# -----------------------------------------------------------------------------
# HDF helpers


def test_normalize_hdf_key():
    assert nocte.core.hdf.normalize_hdf_key('foo') == 'foo'
    assert nocte.core.hdf.normalize_hdf_key('/foo') == 'foo'
    assert nocte.core.hdf.normalize_hdf_key('/foo/bar/') == 'foo/bar'

    with pytest.raises(ValueError, match='cannot be empty'):
        nocte.core.hdf.normalize_hdf_key('')

    with pytest.raises(ValueError, match='cannot be empty'):
        nocte.core.hdf.normalize_hdf_key('/')


def test_hdf_scalar_helpers():
    assert nocte.core.hdf.hdf_attr_as_str('abc') == 'abc'
    assert nocte.core.hdf.hdf_attr_as_str(b'abc') == 'abc'

    version = nocte.core.hdf.get_nocte_version()

    assert isinstance(version, str)
    assert version


def test_prepare_hdf_key_overwrite(tmp_path: pathlib.Path):
    path = tmp_path / 'test.h5'

    with h5py.File(path, mode='w') as file:
        file.create_group('target')
        file.create_group('keep')

    with pytest.raises(FileExistsError):
        nocte.core.hdf.prepare_hdf_key(
            path,
            'target',
            overwrite=False,
        )

    key = nocte.core.hdf.prepare_hdf_key(
        path,
        '/target/',
        overwrite=True,
    )

    assert key == 'target'

    with h5py.File(path, mode='r') as file:
        assert 'target' not in file
        assert 'keep' in file


def test_collection_hdf_attrs(tmp_path: pathlib.Path):
    path = tmp_path / 'test.h5'

    with h5py.File(path, mode='w') as file:
        file.create_group('object')

    nocte.core.hdf.write_hdf_collection_attrs(
        path,
        'object',
        kind='traces',
    )

    with h5py.File(path, mode='r') as file:
        node = file['object']

        assert isinstance(node, h5py.Group)
        assert nocte.core.hdf.hdf_attr_as_str(node.attrs['kind']) == 'traces'
        assert 'nocte_version' in node.attrs

    key = nocte.core.hdf.check_hdf_collection_attrs(
        path,
        '/object/',
        expected_kind='traces',
    )

    assert key == 'object'

    with pytest.raises(ValueError, match='expected'):
        nocte.core.hdf.check_hdf_collection_attrs(
            path,
            'object',
            expected_kind='windows',
        )


def test_collection_hdf_attrs_warn_on_version_mismatch(
    tmp_path: pathlib.Path,
):
    path = tmp_path / 'test.h5'

    with h5py.File(path, mode='w') as file:
        group = file.create_group('object')
        group.attrs['kind'] = 'traces'
        group.attrs['nocte_version'] = 'definitely-not-this-version'

    with pytest.warns(UserWarning):
        key = nocte.core.hdf.check_hdf_collection_attrs(
            path,
            'object',
            expected_kind='traces',
        )

    assert key == 'object'


# -----------------------------------------------------------------------------
# Traces


def test_traces_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'traces.h5'

    values = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=np.float32,
    )

    meta = pd.DataFrame(
        {
            'animal': ['a', 'b'],
            'channel': [3, 7],
        },
        index=pd.Index(
            [10, 20],
            name='trace_id',
        ),
    )

    traces = nocte.core.traces.Traces.from_array(
        values,
        hz=2_000,
        start=-12.5,
        meta=meta,
    )

    traces.to_hdf(
        path,
        key='test',
    )

    loaded = nocte.core.traces.Traces.from_hdf(
        path,
        key='test',
    )

    pd.testing.assert_frame_equal(
        loaded.meta,
        traces.meta,
    )

    np.testing.assert_allclose(
        loaded.values,
        traces.values,
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    assert loaded.dtype == traces.dtype
    assert loaded.shape == traces.shape
    assert loaded.hz == traces.hz
    assert loaded.start == traces.start


def test_empty_traces_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'traces_empty.h5'

    traces = nocte.core.traces.Traces.from_array(
        np.empty(
            (0, 0),
            dtype=np.float32,
        ),
        hz=500,
        start=-10,
    )

    traces.to_hdf(path)

    loaded = nocte.core.traces.Traces.from_hdf(path)

    assert len(loaded) == 0
    assert loaded.shape == (0, 0)
    assert loaded.dtype == np.dtype(np.float32)
    assert loaded.hz == 500
    assert loaded.start == -10

    pd.testing.assert_frame_equal(
        loaded.meta,
        traces.meta,
    )


# -----------------------------------------------------------------------------
# Windows


def test_windows_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'windows.h5'

    meta = pd.DataFrame(
        {
            'state': ['rem', 'sws', 'rem'],
            'animal': ['a', 'a', 'b'],
        },
        index=pd.Index(
            [10, 20, 40],
            name='win_id',
        ),
    )

    windows = nocte.core.windows.Windows.from_arrays(
        start=[-10, 0, -5],
        stop=[20, 100, 15],
        ref=[1_000, 2_000, 3_000],
        meta=meta,
    )

    windows.to_hdf(
        path,
        key='test',
    )

    loaded = nocte.core.windows.Windows.from_hdf(
        path,
        key='test',
    )

    pd.testing.assert_frame_equal(
        loaded.meta,
        windows.meta,
    )

    pd.testing.assert_series_equal(
        loaded.start,
        windows.start,
    )

    pd.testing.assert_series_equal(
        loaded.stop,
        windows.stop,
    )

    pd.testing.assert_series_equal(
        loaded.ref,
        windows.ref,
    )


def test_empty_windows_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'windows_empty.h5'

    windows = nocte.core.windows.Windows.from_arrays(
        [],
        [],
        [],
    )

    windows.to_hdf(path)

    loaded = nocte.core.windows.Windows.from_hdf(path)

    assert len(loaded) == 0

    pd.testing.assert_frame_equal(
        loaded.meta,
        windows.meta,
    )

    pd.testing.assert_series_equal(
        loaded.start,
        windows.start,
    )

    pd.testing.assert_series_equal(
        loaded.stop,
        windows.stop,
    )

    pd.testing.assert_series_equal(
        loaded.ref,
        windows.ref,
    )
