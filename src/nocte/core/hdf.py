import importlib.metadata
import pathlib
import typing
import warnings

import h5py


class SerializableCollection(typing.Protocol):
    """Object with self-contained HDF serialization."""

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
        overwrite: bool = False,
    ) -> None: ...

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> typing.Self: ...


def normalize_hdf_key(key: str) -> str:
    key = key.strip('/')

    if not key:
        raise ValueError('HDF5 key cannot be empty')

    return key


def get_nocte_version() -> str:
    try:
        return importlib.metadata.version('nocte')
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'


def hdf_attr_as_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()

    return str(value)


def prepare_hdf_key(
    path: str | pathlib.Path,
    key: str,
    *,
    overwrite: bool,
) -> str:
    """
    Prepare a collection root for writing.

    If the key already exists and overwrite is False, raise FileExistsError.
    If overwrite is True, remove the complete existing subtree.

    The key itself is not created here.
    """
    key = normalize_hdf_key(key)
    path = pathlib.Path(path)

    if not path.exists():
        return key

    with h5py.File(path, mode='a') as file:
        if key not in file:
            return key

        if not overwrite:
            raise FileExistsError(f'HDF5 key {key!r} already exists in {path}')

        del file[key]

    return key


def write_hdf_collection_attrs(
    path: str | pathlib.Path,
    key: str,
    *,
    kind: str,
) -> None:
    """Write standard nocte collection attributes to an existing root."""
    key = normalize_hdf_key(key)

    with h5py.File(path, mode='a') as file:
        if key not in file:
            raise KeyError(f'HDF5 key {key!r} does not exist')

        node = file[key]

        if not isinstance(node, h5py.Group):
            raise TypeError(f'HDF5 collection root {key!r} must be a group')

        node.attrs['kind'] = kind
        node.attrs['nocte_version'] = get_nocte_version()


def check_hdf_collection_attrs(
    path: str | pathlib.Path,
    key: str,
    *,
    expected_kind: str,
) -> str:
    """
    Validate standard collection attributes before loading.

    A wrong collection kind is an error. A missing or different nocte
    version emits a warning and loading continues.

    Returns the normalized key.
    """
    key = normalize_hdf_key(key)

    with h5py.File(path, mode='r') as file:
        if key not in file:
            raise KeyError(f'HDF5 key {key!r} does not exist')

        node = file[key]

        if not isinstance(node, h5py.Group):
            raise TypeError(f'HDF5 collection root {key!r} must be a group')

        stored_kind_raw = node.attrs.get('kind')
        stored_version_raw = node.attrs.get('nocte_version')

    if stored_kind_raw is None:
        raise ValueError(f'HDF5 collection {key!r} is missing its kind attribute')

    stored_kind = hdf_attr_as_str(stored_kind_raw)

    if stored_kind != expected_kind:
        raise ValueError(
            f'HDF5 collection {key!r} has kind {stored_kind!r}; '
            f'expected {expected_kind!r}'
        )

    current_version = get_nocte_version()

    if stored_version_raw is None:
        warnings.warn(
            f'HDF5 collection {key!r} does not record a nocte version; '
            'attempting to load it anyway.',
            UserWarning,
            stacklevel=2,
        )

        return key

    stored_version = hdf_attr_as_str(stored_version_raw)

    if stored_version != current_version:
        warnings.warn(
            f'HDF5 collection {key!r} was written with nocte '
            f'{stored_version!r}, but the current version is '
            f'{current_version!r}; attempting to load it anyway.',
            UserWarning,
            stacklevel=2,
        )

    return key
