import datetime
import pathlib

import h5py
import numpy as np
import pandas as pd
import pytest

import nocte._coll.frames
import nocte._coll.traces
import nocte._coll.windows
import nocte._core.grouping
import nocte._core.hdf

# -----------------------------------------------------------------------------
# HDF helpers


def test_normalize_hdf_key():
    assert nocte._core.hdf.normalize_hdf_key('foo') == 'foo'
    assert nocte._core.hdf.normalize_hdf_key('/foo') == 'foo'
    assert nocte._core.hdf.normalize_hdf_key('/foo/bar/') == 'foo/bar'

    with pytest.raises(ValueError, match='cannot be empty'):
        nocte._core.hdf.normalize_hdf_key('')

    with pytest.raises(ValueError, match='cannot be empty'):
        nocte._core.hdf.normalize_hdf_key('/')


def test_hdf_scalar_helpers():
    assert nocte._core.hdf.hdf_attr_as_str('abc') == 'abc'
    assert nocte._core.hdf.hdf_attr_as_str(b'abc') == 'abc'

    version = nocte._core.hdf.get_nocte_version()

    assert isinstance(version, str)
    assert version


def test_prepare_hdf_key_overwrite(tmp_path: pathlib.Path):
    path = tmp_path / 'test.h5'

    with h5py.File(path, mode='w') as file:
        file.create_group('target')
        file.create_group('keep')

    with pytest.raises(FileExistsError):
        nocte._core.hdf.prepare_hdf_key(
            path,
            'target',
            overwrite=False,
        )

    key = nocte._core.hdf.prepare_hdf_key(
        path,
        '/target/',
        overwrite=True,
    )

    assert key == 'target'

    with h5py.File(path, mode='r') as file:
        assert 'target' not in file
        assert 'keep' in file


def test_get_hdf_save_timestamp():
    timestamp = nocte._core.hdf.get_hdf_save_timestamp()

    assert isinstance(timestamp, str)
    assert datetime.datetime.fromisoformat(timestamp).tzinfo is not None


def test_collection_hdf_attrs(tmp_path: pathlib.Path):
    path = tmp_path / 'test.h5'

    with h5py.File(path, mode='w') as file:
        file.create_group('object')

    before = datetime.datetime.now(datetime.timezone.utc)

    info = nocte._core.hdf.HDFCollectionInfo.new(kind='traces')
    info.to_hdf(path, 'object')

    after = datetime.datetime.now(datetime.timezone.utc)

    with h5py.File(path, mode='r') as file:
        node = file['object']

        assert isinstance(node, h5py.Group)
        assert nocte._core.hdf.hdf_attr_as_str(node.attrs['kind']) == 'traces'
        assert 'nocte_version' in node.attrs

        timestamp = datetime.datetime.fromisoformat(
            nocte._core.hdf.hdf_attr_as_str(node.attrs['timestamp'])
        )

        assert before <= timestamp <= after

    info = nocte._core.hdf.HDFCollectionInfo.from_hdf(
        path,
        '/object/',
    )

    info.validate(
        key='object',
        expected_kind='traces',
    )

    with pytest.raises(ValueError, match='expected'):
        info.validate(
            key='object',
            expected_kind='windows',
        )


def test_hdf_collection_info_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'test.h5'

    with h5py.File(path, mode='w') as file:
        file.create_group('object')

    info = nocte._core.hdf.HDFCollectionInfo.new(kind='traces')
    info.to_hdf(path, 'object')

    info = nocte._core.hdf.HDFCollectionInfo.from_hdf(path, 'object')

    assert info.kind == 'traces'
    assert info.nocte_version == nocte._core.hdf.get_nocte_version()
    assert isinstance(info.timestamp, str)
    assert datetime.datetime.fromisoformat(info.timestamp).tzinfo is not None


def test_hdf_collection_info_missing_attrs(tmp_path: pathlib.Path):
    path = tmp_path / 'test.h5'

    with h5py.File(path, mode='w') as file:
        file.create_group('object')

    with pytest.raises(KeyError, match='missing attributes'):
        nocte._core.hdf.HDFCollectionInfo.from_hdf(path, 'object')


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

    traces = nocte._coll.traces.Traces.from_array(
        values,
        hz=2_000,
        start=-12.5,
        meta=meta,
    )

    traces.to_hdf(
        path,
        key='test',
    )

    loaded = nocte._coll.traces.Traces.from_hdf(
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


def test_hdf_collection_hdf_info(tmp_path: pathlib.Path):
    path = tmp_path / 'traces.h5'

    traces = nocte._coll.traces.Traces.from_array(
        np.array([[1.0, 2.0]], dtype=np.float32),
        hz=100,
        start=0,
    )

    traces.to_hdf(path, key='test')

    info = nocte._coll.traces.Traces.hdf_info(path, key='test')

    assert info.kind == 'traces'
    assert info.nocte_version == nocte._core.hdf.get_nocte_version()
    assert isinstance(info.timestamp, str)


def test_empty_traces_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'traces_empty.h5'

    traces = nocte._coll.traces.Traces.from_array(
        np.empty(
            (0, 0),
            dtype=np.float32,
        ),
        hz=500,
        start=-10,
    )

    traces.to_hdf(path)

    loaded = nocte._coll.traces.Traces.from_hdf(path)

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

    windows = nocte._coll.windows.Windows.from_arrays(
        start=[-10, 0, -5],
        stop=[20, 100, 15],
        ref=[1_000, 2_000, 3_000],
        meta=meta,
    )

    windows.to_hdf(
        path,
        key='test',
    )

    loaded = nocte._coll.windows.Windows.from_hdf(
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

    windows = nocte._coll.windows.Windows.from_arrays(
        [],
        [],
        [],
    )

    windows.to_hdf(path)

    loaded = nocte._coll.windows.Windows.from_hdf(path)

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


# -----------------------------------------------------------------------------
# Frames


def test_frames_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'frames.h5'

    frame0 = pd.DataFrame(
        {
            'x': [1.0, 2.0, np.nan],
            'y': [10, 20, 30],
        },
        index=pd.Index(
            [100, 200, 300],
            name='time',
        ),
    )

    frame1 = pd.DataFrame(
        {
            'value': [4.0, 5.0],
            'label': ['a', 'b'],
        },
        index=pd.Index(
            [7, 9],
            name='sample',
        ),
    )

    meta = pd.DataFrame(
        {
            'animal': ['a', 'b'],
            'channel': [3, 7],
        },
        index=pd.Index(
            [10, 20],
            name='frame_id',
        ),
    )

    frames = nocte._coll.frames.Frames.from_items(
        [frame0, frame1],
        meta=meta,
    )

    frames.to_hdf(
        path,
        key='test',
    )

    loaded = nocte._coll.frames.Frames.from_hdf(
        path,
        key='test',
    )

    pd.testing.assert_frame_equal(
        loaded.meta,
        frames.meta,
    )

    assert len(loaded) == len(frames)

    for idx in frames.index:
        pd.testing.assert_frame_equal(
            loaded.get(idx),
            frames.get(idx),
        )


def test_empty_frames_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'frames_empty.h5'

    frames = nocte._coll.frames.Frames.from_items(
        [],
    )

    frames.to_hdf(path)

    loaded = nocte._coll.frames.Frames.from_hdf(path)

    assert len(loaded) == 0

    pd.testing.assert_frame_equal(
        loaded.meta,
        frames.meta,
    )


# -----------------------------------------------------------------------------
# Grouping


def test_grouping_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'grouping.h5'

    traces0 = nocte._coll.traces.Traces.from_array(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        hz=1_000,
        start=0,
        meta=pd.DataFrame(
            {
                'channel': [1, 2],
            },
            index=pd.Index(
                [10, 20],
                name='trace_id',
            ),
        ),
    )

    traces1 = nocte._coll.traces.Traces.from_array(
        np.array(
            [
                [10.0, 20.0],
            ],
            dtype=np.float32,
        ),
        hz=1_000,
        start=100,
        meta=pd.DataFrame(
            {
                'channel': [7],
            },
            index=pd.Index(
                [30],
                name='trace_id',
            ),
        ),
    )

    meta = pd.DataFrame(
        {
            'animal': ['a', 'b'],
            'condition': ['control', 'stim'],
        },
        index=pd.Index(
            [100, 200],
            name='group_id',
        ),
    )

    grouping = nocte._core.grouping.Grouping.from_items(
        [traces0, traces1],
        meta=meta,
    )

    grouping.to_hdf(
        path,
        key='test',
    )

    loaded = nocte._core.grouping.Grouping.from_hdf(
        path,
        item_type=nocte._coll.traces.Traces,
        key='test',
    )

    pd.testing.assert_frame_equal(
        loaded.meta,
        grouping.meta,
    )

    assert len(loaded) == len(grouping)

    for idx in grouping.index:
        expected = grouping.get(idx)
        actual = loaded.get(idx)

        pd.testing.assert_frame_equal(
            actual.meta,
            expected.meta,
        )

        np.testing.assert_allclose(
            actual.values,
            expected.values,
            rtol=0,
            atol=0,
            equal_nan=True,
        )

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.hz == expected.hz
        assert actual.start == expected.start


def test_empty_grouping_hdf_roundtrip(tmp_path: pathlib.Path):
    path = tmp_path / 'grouping_empty.h5'

    grouping = nocte._core.grouping.Grouping.from_items(
        [],
    )

    grouping.to_hdf(
        path,
        key='test',
    )

    loaded = nocte._core.grouping.Grouping.from_hdf(
        path,
        item_type=nocte._coll.traces.Traces,
        key='test',
    )

    assert len(loaded) == 0

    pd.testing.assert_frame_equal(
        loaded.meta,
        grouping.meta,
    )
