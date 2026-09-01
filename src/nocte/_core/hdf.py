import abc
import datetime
import importlib.metadata
import pathlib
import typing
import warnings

import h5py
import pandas as pd

from nocte._core.collection import Collection, ItemT, PBarParamT


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


def get_hdf_save_timestamp() -> str:
    """Current UTC time as an ISO 8601 string, for provenance."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def hdf_attr_as_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()

    return str(value)


class HDFCollectionInfo(typing.NamedTuple):
    """Provenance attributes stored alongside a collection root."""

    kind: str
    nocte_version: str
    timestamp: str

    @classmethod
    def new(cls, *, kind: str) -> typing.Self:
        return cls(
            kind=kind,
            nocte_version=get_nocte_version(),
            timestamp=get_hdf_save_timestamp(),
        )

    def to_hdf(
        self,
        path: str | pathlib.Path,
        key: str,
    ) -> None:
        """Write these attributes to an existing collection root."""
        key = normalize_hdf_key(key)

        with h5py.File(path, mode='a') as file:
            node = _require_hdf_group(file, key)

            node.attrs['kind'] = self.kind
            node.attrs['nocte_version'] = self.nocte_version
            node.attrs['timestamp'] = self.timestamp

    @classmethod
    def from_hdf(
        cls,
        path: str | pathlib.Path,
        key: str,
    ) -> typing.Self:
        """Read the attributes stored at a collection root."""
        key = normalize_hdf_key(key)

        with h5py.File(path, mode='r') as file:
            node = _require_hdf_group(file, key)

            missing = {'kind', 'nocte_version', 'timestamp'} - node.attrs.keys()

            if missing:
                raise KeyError(
                    f'HDF5 collection {key!r} is missing attributes: {sorted(missing)}'
                )

            kind = hdf_attr_as_str(node.attrs['kind'])
            nocte_version = hdf_attr_as_str(node.attrs['nocte_version'])
            timestamp = hdf_attr_as_str(node.attrs['timestamp'])

        return cls(kind=kind, nocte_version=nocte_version, timestamp=timestamp)

    def validate(self, key: str, expected_kind: str):
        """
        Validate stored attributes.

        A wrong kind is an error. A different nocte version emits a warning
        and loading continues. Returns the normalized key.
        """
        if self.kind != expected_kind:
            raise ValueError(
                f'HDF5 collection {key!r} has kind {self.kind!r}; '
                f'expected {expected_kind!r}'
            )

        current_version = get_nocte_version()

        if self.nocte_version != current_version:
            warnings.warn(
                f'HDF5 collection {key!r} was written with nocte '
                f'{self.nocte_version!r}, but the current version is '
                f'{current_version!r}; attempting to load it anyway.',
                UserWarning,
                stacklevel=2,
            )


def _require_hdf_group(
    file: h5py.File,
    key: str,
) -> h5py.Group:
    if key not in file:
        raise KeyError(f'HDF5 key {key!r} does not exist')

    node = file[key]

    if not isinstance(node, h5py.Group):
        raise TypeError(f'HDF5 collection root {key!r} must be a group')

    return node


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

        info = HDFCollectionInfo.new(kind=self._hdf_kind())

        info.to_hdf(path, key)

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

        info = HDFCollectionInfo.from_hdf(path, key)

        info.validate(key=key, expected_kind=cls._hdf_kind())

        meta = pd.read_hdf(
            path,
            key=f'{key}/meta',
        )

        if not isinstance(meta, pd.DataFrame):
            raise TypeError(f'HDF metadata at {key!r}/meta is not a DataFrame')

        return cls._from_hdf_data(path, key=key, meta=meta)

    @classmethod
    def hdf_info(
        cls,
        path: str | pathlib.Path,
        *,
        key: str | None = None,
    ) -> HDFCollectionInfo:
        """Read stored provenance attributes without loading the collection."""
        if key is None:
            key = cls._hdf_kind()

        return HDFCollectionInfo.from_hdf(path, key)

    @classmethod
    def from_hdf_grouping(
        cls,
        path: str | pathlib.Path,
        *,
        key: str = 'grouping',
        pbar: PBarParamT = False,
    ):
        """Load a grouping containing collections of this type."""
        import nocte._core.grouping

        return nocte._core.grouping.Grouping.from_hdf(
            path,
            item_type=cls,
            key=key,
            pbar=pbar,
        )
