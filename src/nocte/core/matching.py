from __future__ import annotations

import collections.abc
import typing

import numpy as np
import pandas as pd

from nocte.core.collection import Collection, IndexLike, as_ids, is_valid_name

Match = tuple[int, int]


def _normalize_by(
    by: str | collections.abc.Sequence[str] | None,
) -> tuple[str, ...] | None:
    if by is None:
        return None

    columns = (by,) if isinstance(by, str) else tuple(by)

    if not columns:
        raise ValueError('by must contain at least one metadata column')

    if any(not isinstance(column, str) for column in columns):
        raise TypeError('matching columns must be strings')

    if len(set(columns)) != len(columns):
        raise ValueError('matching columns must be unique')

    return columns


class _MatchesData:
    """Internal positional storage for matched source identities."""

    def __init__(
        self,
        left: pd.Index | np.ndarray,
        right: pd.Index | np.ndarray,
    ):
        left = np.asarray(left)
        right = np.asarray(right)

        if left.ndim != 1 or right.ndim != 1:
            raise ValueError('match ID arrays must be one-dimensional')

        if len(left) != len(right):
            raise ValueError('left and right match ID arrays must have equal length')

        if not np.issubdtype(
            left.dtype,
            np.integer,
        ):
            raise TypeError('left match IDs must contain integers')

        if not np.issubdtype(
            right.dtype,
            np.integer,
        ):
            raise TypeError('right match IDs must contain integers')

        left = np.array(
            left,
            copy=True,
            order='C',
            subok=False,
        )

        right = np.array(
            right,
            copy=True,
            order='C',
            subok=False,
        )

        left.flags.writeable = False
        right.flags.writeable = False

        self._left: np.ndarray = left
        self._right: np.ndarray = right

    def __len__(self) -> int:
        assert len(self._left) == len(self._right)
        return len(self._left)

    @property
    def left(self) -> np.ndarray:
        return self._left

    @property
    def right(self) -> np.ndarray:
        return self._right

    def get_pos(
        self,
        position: int,
    ) -> Match:
        return (
            int(self._left[position]),
            int(self._right[position]),
        )

    def sel_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        positions = np.asarray(
            positions,
            dtype=np.intp,
        )

        return self.__class__(
            self._left[positions],
            self._right[positions],
        )

    def copy(self) -> typing.Self:
        return self


class Matches(
    Collection[Match],
):
    """
    Ordered relation between items of two Collections.

    Each item is one match with its own unique identity.

    The payload stores the matched source IDs. Metadata stores optional
    pair-specific information.

    The source Collections themselves are not retained. Their identity
    namespaces are recorded through ``left_name`` and ``right_name``.
    """

    def __init__(
        self,
        data: _MatchesData,
        meta: pd.DataFrame,
        *,
        left_name: str,
        right_name: str,
        by: str | collections.abc.Sequence[str] | None = None,
    ):
        if not isinstance(
            data,
            _MatchesData,
        ):
            raise TypeError('data must be _MatchesData')

        if not is_valid_name(left_name):
            raise ValueError('left_name must be a non-empty string')

        if not is_valid_name(right_name):
            raise ValueError('right_name must be a non-empty string')

        self._data = data
        self.meta = meta.copy()
        self._left_name = left_name
        self._right_name = right_name
        self._by = _normalize_by(by)

        self._validate_meta(len(self._data))

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def _build(
        cls,
        left_ids: IndexLike,
        right_ids: IndexLike,
        left_name: str,
        right_name: str,
        *,
        meta: pd.DataFrame | None,
        name: str,
        by: str | collections.abc.Sequence[str] | None,
    ) -> typing.Self:

        left_ids = as_ids(left_ids, side='left')
        right_ids = as_ids(right_ids, side='right')

        if len(left_ids) != len(right_ids):
            raise ValueError('left_ids and right_ids must have the same length')

        data = _MatchesData(
            left_ids.to_numpy(copy=False),
            right_ids.to_numpy(copy=False),
        )

        if meta is None:
            meta = cls._default_meta(
                len(data),
                name=name,
            )

        else:
            meta = meta.copy()

            if len(meta) != len(data):
                raise ValueError(
                    f'meta has {len(meta)} rows, but there are {len(data)} matches'
                )

            meta.index = meta.index.rename(name)

        return cls(
            data,
            meta,
            left_name=left_name,
            right_name=right_name,
            by=by,
        )

    @classmethod
    def from_meta(
        cls,
        left: Collection[typing.Any],
        right: Collection[typing.Any],
        *,
        by: str | collections.abc.Sequence[str],
        name: str = 'match',
    ) -> typing.Self:
        """Construct the many-to-many equality relation on metadata columns."""
        columns = _normalize_by(by)

        if columns is None:
            raise RuntimeError('normalized matching columns cannot be None')

        missing_left = set(columns).difference(left.meta.columns)
        missing_right = set(columns).difference(right.meta.columns)

        if missing_left:
            raise KeyError(
                f'left collection is missing metadata columns: {sorted(missing_left)}'
            )

        if missing_right:
            raise KeyError(
                f'right collection is missing metadata columns: {sorted(missing_right)}'
            )

        if left.name == right.name:
            left_id_col = f'left_{left.name}'
            right_id_col = f'right_{right.name}'
        else:
            left_id_col = left.name
            right_id_col = right.name

        if left_id_col in columns:
            raise ValueError(
                f'left identity name {left_id_col!r} conflicts with a matching column'
            )

        if right_id_col in columns:
            raise ValueError(
                f'right identity name {right_id_col!r} conflicts with a matching column'
            )

        left_frame = left.meta.loc[:, list(columns)].copy()
        right_frame = right.meta.loc[:, list(columns)].copy()

        left_frame[left_id_col] = left.index.to_numpy(copy=False)
        right_frame[right_id_col] = right.index.to_numpy(copy=False)

        merged = left_frame.merge(
            right_frame,
            how='inner',
            on=list(columns),
            sort=False,
            validate='many_to_many',
        )

        return cls._build(
            left_name=left.name,
            right_name=right.name,
            left_ids=pd.Index(merged[left_id_col]),
            right_ids=pd.Index(merged[right_id_col]),
            meta=None,
            name=name,
            by=columns,
        )

    @classmethod
    def from_product(
        cls,
        left: Collection[typing.Any],
        right: Collection[typing.Any],
        *,
        name: str = 'match',
    ) -> typing.Self:
        """Construct the Cartesian product of two collections."""
        left_ids = left.index.to_numpy(copy=False)

        right_ids = right.index.to_numpy(copy=False)

        return cls._build(
            left_name=left.name,
            right_name=right.name,
            left_ids=pd.Index(
                np.repeat(
                    left_ids,
                    len(right),
                )
            ),
            right_ids=pd.Index(
                np.tile(
                    right_ids,
                    len(left),
                )
            ),
            meta=None,
            name=name,
            by=None,
        )

    @classmethod
    def from_identity(
        cls,
        collection: Collection[typing.Any],
        *,
        name: str = 'match',
    ) -> typing.Self:
        """Construct the one-to-one self relation."""
        return cls._build(
            left_name=collection.name,
            right_name=collection.name,
            left_ids=collection.index,
            right_ids=collection.index,
            meta=None,
            name=name,
            by=None,
        )

    @classmethod
    def from_combinations(
        cls,
        collection: Collection[typing.Any],
        *,
        name: str = 'match',
    ) -> typing.Self:
        """Construct every unique unordered pair of distinct items."""
        left_pos, right_pos = np.triu_indices(
            len(collection),
            k=1,
        )

        ids = collection.index.to_numpy(copy=False)

        return cls._build(
            left_name=collection.name,
            right_name=collection.name,
            left_ids=pd.Index(ids[left_pos]),
            right_ids=pd.Index(ids[right_pos]),
            meta=None,
            name=name,
            by=None,
        )

    # ------------------------------------------------------------------
    # relation state

    @property
    def left_name(self) -> str:
        return self._left_name

    @property
    def right_name(self) -> str:
        return self._right_name

    @property
    def by(self) -> tuple[str, ...] | None:
        return self._by

    def _source_column_names(
        self,
    ) -> tuple[str, str]:
        if self.left_name == self.right_name:
            return (
                f'left_{self.left_name}',
                f'right_{self.right_name}',
            )

        return (
            self.left_name,
            self.right_name,
        )

    @property
    def left(self) -> pd.Series:
        """Left source IDs indexed by match identity."""
        left_name, _ = self._source_column_names()

        return pd.Series(
            self._data.left.copy(),
            index=self.index.copy(),
            name=left_name,
        )

    @property
    def right(self) -> pd.Series:
        """Right source IDs indexed by match identity."""
        _, right_name = self._source_column_names()

        return pd.Series(
            self._data.right.copy(),
            index=self.index.copy(),
            name=right_name,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return source identities as a match-indexed DataFrame."""
        return pd.concat(
            [
                self.left,
                self.right,
            ],
            axis=1,
        )

    # ------------------------------------------------------------------
    # source resolution

    def _positions_in(
        self,
        source: Collection[typing.Any],
        ids: np.ndarray,
        *,
        expected_name: str,
        side: str,
    ) -> np.ndarray:
        if source.name != expected_name:
            raise ValueError(
                f'{side} source has name '
                f'{source.name!r}, but Matches '
                f'expects {expected_name!r}'
            )

        positions = source.index.get_indexer(pd.Index(ids))

        missing = positions < 0

        if missing.any():
            missing_ids = pd.Index(ids[missing]).unique().tolist()

            raise KeyError(f'{side} source is missing matched IDs: {missing_ids}')

        return positions

    def left_positions(
        self,
        source: Collection[typing.Any],
    ) -> np.ndarray:
        """
        Resolve repeated left source IDs to positions.

        Unlike Collection._positions(), repeated source identities are
        expected and preserved.
        """
        return self._positions_in(
            source,
            self._data.left,
            expected_name=self.left_name,
            side='left',
        )

    def right_positions(
        self,
        source: Collection[typing.Any],
    ) -> np.ndarray:
        """
        Resolve repeated right source IDs to positions.

        Unlike Collection._positions(), repeated source identities are
        expected and preserved.
        """
        return self._positions_in(
            source,
            self._data.right,
            expected_name=self.right_name,
            side='right',
        )

    # ------------------------------------------------------------------
    # relation lookup

    def left_for(
        self,
        right_id: int,
    ) -> pd.Index:
        """Return left IDs related to one right ID."""
        return pd.Index(
            self._data.left[self._data.right == right_id],
            name=self.left_name,
        )

    def right_for(
        self,
        left_id: int,
    ) -> pd.Index:
        """Return right IDs related to one left ID."""
        return pd.Index(
            self._data.right[self._data.left == left_id],
            name=self.right_name,
        )

    def iter_left(
        self,
    ) -> collections.abc.Iterator[tuple[int, pd.Index]]:
        """
        Iterate represented left IDs and their related right IDs.

        IDs follow first appearance in the relation.
        """
        for left_id in pd.unique(self._data.left):
            yield (
                int(left_id),
                self.right_for(int(left_id)),
            )

    def iter_right(
        self,
    ) -> collections.abc.Iterator[tuple[int, pd.Index]]:
        """
        Iterate represented right IDs and their related left IDs.

        IDs follow first appearance in the relation.
        """
        for right_id in pd.unique(self._data.right):
            yield (
                int(right_id),
                self.left_for(int(right_id)),
            )

    def _side_meta(
        self,
        source: Collection[typing.Any],
        *,
        side: typing.Literal['left', 'right'],
    ) -> pd.DataFrame:
        """
        Materialize source and match metadata with one row per match.

        The result uses match identities as its index and preserves the source
        identity as provenance.
        """
        if side == 'left':
            expected_name = self.left_name
            source_ids = self.left.to_numpy(copy=False)
            positions = self.left_positions(source)
        else:
            expected_name = self.right_name
            source_ids = self.right.to_numpy(copy=False)
            positions = self.right_positions(source)

        if source.name != expected_name:
            raise ValueError(
                f'source does not correspond to the {side} side of Matches: '
                f'expected {expected_name!r}, got {source.name!r}'
            )

        if self.left_name == self.right_name:
            source_id_name = f'{side}_{source.name}'
        else:
            source_id_name = source.name

        meta = source.meta.iloc[positions].copy()

        # Preserve source identity as provenance.
        if source_id_name in meta.columns:
            existing = meta[source_id_name].to_numpy(copy=False)

            if not np.array_equal(
                existing,
                source_ids,
            ):
                raise ValueError(
                    f'source metadata column {source_id_name!r} '
                    'contradicts source identity'
                )
        else:
            meta[source_id_name] = source_ids

        # The derived items are matches.
        meta.index = self.index

        # Match-specific metadata belongs to the derived items too.
        for column in self.meta.columns:
            incoming = self.meta[column]

            if column not in meta.columns:
                meta[column] = incoming
                continue

            existing = meta[column]

            same = (existing.eq(incoming) | (existing.isna() & incoming.isna())).fillna(
                False
            )

            if not same.all():
                raise ValueError(
                    f'match metadata column {column!r} contradicts source metadata'
                )

        return meta

    def left_meta(
        self,
        source: Collection[typing.Any],
    ) -> pd.DataFrame:
        """
        Materialize left-source and match metadata with one row per match.
        """
        return self._side_meta(
            source,
            side='left',
        )

    def right_meta(
        self,
        source: Collection[typing.Any],
    ) -> pd.DataFrame:
        """
        Materialize right-source and match metadata with one row per match.
        """
        return self._side_meta(
            source,
            side='right',
        )

    # ------------------------------------------------------------------
    # collection primitives

    def _sel_pos(
        self,
        positions: np.ndarray,
    ) -> typing.Self:
        return self.__class__(
            self._data.sel_pos(positions),
            self.meta.iloc[positions],
            left_name=self.left_name,
            right_name=self.right_name,
            by=self.by,
        )

    def _get_pos(
        self,
        position: int,
    ) -> Match:
        return self._data.get_pos(position)

    def copy(self) -> typing.Self:
        return self.__class__(
            self._data.copy(),
            self.meta.copy(),
            left_name=self.left_name,
            right_name=self.right_name,
            by=self.by,
        )
