import abc
import importlib.metadata
import pathlib
import typing
import warnings

import h5py
import pandas as pd

from nocte.core.collection import Collection, ItemT, PBarParamT


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


class HDFCollection(Collection[ItemT], abc.ABC):
    """
    Collection using the standard nocte HDF storage envelope.

    Subclasses only define how their payload is written and how a complete
    collection is reconstructed from its stored payload. Metadata and the
    collection-level HDF structure are handled here.
    """

    # ------------------------------------------------------------------------------
    # abstract methods

    @classmethod
    def _hdf_kind(cls) -> str:
        """Stable HDF kind used for this collection class."""
        return cls.__name__.lower()

    @abc.abstractmethod
    def _to_hdf_data(
        self,
        path: str | pathlib.Path,
        *,
        key: str,
    ) -> None:
        """Write the collection-specific payload to ``key``."""
        ...

    @classmethod
    @abc.abstractmethod
    def _from_hdf_data(
        cls,
        path: str | pathlib.Path,
        *,
        key: str,
        meta: pd.DataFrame,
    ) -> typing.Self:
        """Read the collection-specific payload and construct the collection."""
        ...

    # ------------------------------------------------------------------------------
    # public serialization methods

    def to_hdf(
        self,
        path: str | pathlib.Path,
        *,
        key: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Write the collection to HDF5."""
        if key is None:
            key = self._hdf_kind()

        key = prepare_hdf_key(path, key, overwrite=overwrite)

        self.meta.to_hdf(path, key=f'{key}/meta', mode='a')

        write_hdf_collection_attrs(path, key, kind=self._hdf_kind())

        self._to_hdf_data(path, key=key)

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        *,
        key: str | None = None,
    ) -> typing.Self:
        """Load a collection from HDF5."""
        if key is None:
            key = cls._hdf_kind()

        key = check_hdf_collection_attrs(
            path,
            key,
            expected_kind=cls._hdf_kind(),
        )

        meta = pd.read_hdf(
            path,
            key=f'{key}/meta',
        )

        if not isinstance(meta, pd.DataFrame):
            raise TypeError(f'HDF metadata at {key!r}/meta is not a DataFrame')

        return cls._from_hdf_data(path, key=key, meta=meta)

    @classmethod
    def from_hdf_grouping(
        cls,
        path: str | pathlib.Path,
        *,
        key: str = 'grouping',
        pbar: PBarParamT = False,
    ):
        """Load a grouping containing collections of this type."""
        import nocte.core.grouping

        return nocte.core.grouping.Grouping.from_hdf(
            path,
            item_type=cls,
            key=key,
            pbar=pbar,
        )
