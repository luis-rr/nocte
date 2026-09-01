# Add near the top of registry.py:
import collections.abc
import dataclasses
import pathlib
import string
import typing

import numpy as np
import pandas as pd

import nocte.core.collection
import nocte.core.events
import nocte.core.windows
from nocte.core.registry import Registry
from nocte.core.sources import Source, SourceKind, Sources, SourcesGrouping

Tag = tuple[str, ...]
TagLike = str | tuple[str, ...]

WindowsParser = collections.abc.Callable[
    [object],
    nocte.core.windows.Windows | None,
]

EventsParser = collections.abc.Callable[
    [object],
    nocte.core.events.Events | None,
]


def _as_tag(tag: TagLike) -> Tag:
    if isinstance(tag, str):
        tag = (tag,)

    if (
        not isinstance(tag, tuple)
        or not tag
        or any(not isinstance(part, str) or not part for part in tag)
    ):
        raise ValueError(
            'tag must be a non-empty string or a non-empty tuple of non-empty strings'
        )

    return tag


def _as_columns(
    columns: str | collections.abc.Sequence[str],
) -> tuple[str, ...]:
    if isinstance(columns, str):
        columns = (columns,)
    else:
        columns = tuple(columns)

    if not columns:
        raise ValueError('at least one column is required')

    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError('column names must be non-empty strings')

    return columns


def _is_missing(value: object) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)

    return False


@dataclasses.dataclass(frozen=True, slots=True)
class _SourceSpec:
    tag: Tag
    path: str
    extra: dict[str, str]
    required: bool
    exists: bool
    kind: SourceKind


@dataclasses.dataclass(frozen=True, slots=True)
class _WindowsSpec:
    tag: Tag
    column: str
    parser: WindowsParser | None
    required: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _EventsSpec:
    tag: Tag
    column: str
    parser: EventsParser | None
    required: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _LiteralRule:
    column: str
    allowed: tuple[object, ...]
    required: bool


class Builder:
    """
    Thin construction helper for project-specific Registry ingestion.

    ``df`` is always the authoritative working table. Ordinary project
    transformations should use pandas directly. Builder only provides the
    common operations needed to clean, validate, and materialize Registry
    resources.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        roots: collections.abc.Sequence[str | pathlib.Path] = (),
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas DataFrame')

        self.df = df.copy()

        # Keep roots as strings so roots written for another operating
        # system retain their original syntax during matching.
        self._roots = tuple(str(root) for root in roots)

        self._required: list[tuple[str, ...]] = []
        self._unique: list[str] = []
        self._together: list[tuple[str, ...]] = []
        self._literals: list[_LiteralRule] = []

        self._source_specs: list[_SourceSpec] = []
        self._windows_specs: list[_WindowsSpec] = []
        self._events_specs: list[_EventsSpec] = []

        self._drop_meta: set[str] = set()
        self._keep_meta: tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # construction

    @classmethod
    def from_excel(
        cls,
        path: str | pathlib.Path,
        *,
        sheet_name: str | int = 0,
        roots: collections.abc.Sequence[str | pathlib.Path] = (),
        drop_empty: bool = True,
    ) -> typing.Self:
        df = pd.read_excel(
            path,
            sheet_name=sheet_name,
        )

        if not isinstance(df, pd.DataFrame):
            raise TypeError('sheet_name must select exactly one Excel sheet')

        if drop_empty:
            df = df.dropna(
                axis=0,
                how='all',
            )
            df = df.dropna(
                axis=1,
                how='all',
            )

        return cls(
            df,
            roots=roots,
        )

    # ------------------------------------------------------------------
    # dataframe transformations

    def drop_rows(
        self,
        mask: nocte.core.collection.MaskLike,
    ) -> None:
        if isinstance(mask, pd.Series):
            if (
                len(mask) != len(self.df)
                or not mask.index.is_unique
                or not self.df.index.difference(mask.index).empty
                or not mask.index.difference(self.df.index).empty
            ):
                raise ValueError('mask index must contain exactly builder.df.index')

            values = mask.reindex(self.df.index).to_numpy()
        else:
            values = np.asarray(mask)

        if values.ndim != 1:
            raise ValueError('mask must be one-dimensional')

        if values.dtype != bool:
            raise TypeError('mask must be boolean')

        if len(values) != len(self.df):
            raise ValueError('mask length does not match builder.df')

        self.df = self.df.loc[~values].copy()

    def fill_down(
        self,
        columns: str | collections.abc.Sequence[str],
    ) -> None:
        columns = _as_columns(columns)

        self._require_columns(columns)

        self.df.loc[
            :,
            list(columns),
        ] = self.df.loc[
            :,
            list(columns),
        ].ffill()

    def derive(
        self,
        name: str,
        *,
        fmt: str,
        required: bool = False,
        unique: bool = False,
    ) -> None:
        """
        Construct a derived column using a missing-aware format template.

        Template fields refer directly to columns in ``builder.df``. If any
        referenced value is missing, the derived value is ``pd.NA``.
        """
        if not isinstance(name, str) or not name:
            raise ValueError('name must be a non-empty string')

        if name in self.df.columns:
            raise ValueError(f'column {name!r} already exists')

        fields = self._template_fields(fmt)
        self._require_columns(fields)

        values: list[object] = []

        for position in range(len(self.df)):
            value = self._format_template(
                fmt,
                position,
                preserve_scalar=False,
            )

            values.append(pd.NA if value is None else value)

        self.df[name] = values

        if required:
            self.require(name)

        if unique:
            self.unique(name)

    # ------------------------------------------------------------------
    # validation

    def require(
        self,
        columns: str | collections.abc.Sequence[str],
    ) -> None:
        columns = _as_columns(columns)

        self._validate_required(columns)
        self._required.append(columns)

    def unique(
        self,
        column: str,
    ) -> None:
        if not isinstance(column, str) or not column:
            raise ValueError('column must be a non-empty string')

        self._validate_unique(column)
        self._unique.append(column)

    def require_together(
        self,
        columns: str | collections.abc.Sequence[str],
    ) -> None:
        columns = _as_columns(columns)

        if len(columns) < 2:
            raise ValueError('require_together needs at least two columns')

        self._validate_together(columns)
        self._together.append(columns)

    def literal(
        self,
        column: str,
        *,
        allowed: collections.abc.Iterable[object],
        required: bool = False,
    ) -> None:
        allowed = tuple(allowed)

        if not allowed:
            raise ValueError('allowed must contain at least one value')

        rule = _LiteralRule(
            column=column,
            allowed=allowed,
            required=required,
        )

        self._validate_literal(rule)
        self._literals.append(rule)

    # ------------------------------------------------------------------
    # resource declarations

    def source(
        self,
        tag: TagLike,
        *,
        path: str,
        extra: collections.abc.Mapping[
            str,
            str,
        ]
        | None = None,
        required: bool = False,
        exists: bool = True,
        kind: SourceKind = 'any',
    ) -> None:
        """
        Declare one Source resource per Registry entry.

        ``path`` and values of ``extra`` are missing-aware format templates.
        The source is absent when its path cannot be produced. Extra values
        are only required when the path is present.
        """
        tag = _as_tag(tag)

        self._require_new_tag(
            tag,
            specs=self._source_specs,
            category='source',
        )

        if kind not in {
            'file',
            'dir',
            'any',
        }:
            raise ValueError("kind must be 'file', 'dir', or 'any'")

        if not isinstance(path, str) or not path:
            raise ValueError('path must be a non-empty format string')

        extra_ = {} if extra is None else dict(extra)

        if any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in extra_.items()
        ):
            raise TypeError('extra must map non-empty string keys to format strings')

        self._source_specs.append(
            _SourceSpec(
                tag=tag,
                path=path,
                extra=extra_,
                required=required,
                exists=exists,
                kind=kind,
            )
        )

    def windows(
        self,
        tag: TagLike,
        *,
        column: str,
        parser: WindowsParser | None = None,
        required: bool = False,
    ) -> None:
        """
        Declare one Windows resource per Registry entry.

        Without ``parser``, non-missing cells must already contain Windows.
        With ``parser``, each non-missing cell is passed to the project parser.
        """
        tag = _as_tag(tag)

        self._require_new_tag(
            tag,
            specs=self._windows_specs,
            category='windows',
        )

        self._windows_specs.append(
            _WindowsSpec(
                tag=tag,
                column=column,
                parser=parser,
                required=required,
            )
        )

    def events(
        self,
        tag: TagLike,
        *,
        column: str,
        parser: EventsParser | None = None,
        required: bool = False,
    ) -> None:
        """
        Declare one Events resource per Registry entry.

        Without ``parser``, non-missing cells must already contain Events.
        With ``parser``, each non-missing cell is passed to the project parser.
        """
        tag = _as_tag(tag)

        self._require_new_tag(
            tag,
            specs=self._events_specs,
            category='events',
        )

        self._events_specs.append(
            _EventsSpec(
                tag=tag,
                column=column,
                parser=parser,
                required=required,
            )
        )

    # ------------------------------------------------------------------
    # final metadata

    def drop_meta(
        self,
        columns: str | collections.abc.Sequence[str],
    ) -> None:
        if self._keep_meta is not None:
            raise ValueError('cannot combine drop_meta() and keep_meta()')

        self._drop_meta.update(_as_columns(columns))

    def keep_meta(
        self,
        columns: str | collections.abc.Sequence[str],
    ) -> None:
        if self._drop_meta:
            raise ValueError('cannot combine keep_meta() and drop_meta()')

        if self._keep_meta is not None:
            raise ValueError('keep_meta() has already been specified')

        self._keep_meta = _as_columns(columns)

    # ------------------------------------------------------------------
    # build

    def build(
        self,
        *,
        name: str,
    ) -> Registry:
        """
        Validate the current dataframe and materialize a clean Registry.

        Registry item IDs are fresh integer IDs. ``name`` identifies the
        semantic namespace of those IDs, for example ``'experiment'`` or
        ``'section'``.
        """
        if not nocte.core.collection.is_valid_name(name):
            raise ValueError('name must be a non-empty string')

        self._validate()

        meta = self._build_meta(name)

        return Registry.from_data(
            meta=meta,
            wins=self._build_windows(name),
            events=self._build_events(name),
            sources=self._build_sources(name),
        )

    # ------------------------------------------------------------------
    # validation internals

    def _validate(
        self,
    ) -> None:
        for columns in self._required:
            self._validate_required(columns)

        for column in self._unique:
            self._validate_unique(column)

        for columns in self._together:
            self._validate_together(columns)

        for rule in self._literals:
            self._validate_literal(rule)

        for spec in self._windows_specs:
            self._require_columns((spec.column,))

        for spec in self._events_specs:
            self._require_columns((spec.column,))

        for spec in self._source_specs:
            self._require_columns(self._template_fields(spec.path))

            for template in spec.extra.values():
                self._require_columns(self._template_fields(template))

    def _validate_required(
        self,
        columns: tuple[str, ...],
    ) -> None:
        self._require_columns(columns)

        for column in columns:
            values = self.df[column]

            bad = [
                self.df.index[position]
                for position in range(len(values))
                if _is_missing(values.iloc[position])
            ]

            if bad:
                raise ValueError(
                    f'column {column!r} is required but is missing at rows {bad}'
                )

    def _validate_unique(
        self,
        column: str,
    ) -> None:
        self._require_columns((column,))

        values = self.df[column]

        present_mask = np.array(
            [not _is_missing(values.iloc[position]) for position in range(len(values))],
            dtype=bool,
        )

        present = values.iloc[np.flatnonzero(present_mask)]

        duplicated = present.duplicated(keep=False)

        if duplicated.any():
            rows = present.index[duplicated].tolist()

            raise ValueError(
                f'column {column!r} must be unique; duplicates occur at rows {rows}'
            )

    def _validate_together(
        self,
        columns: tuple[str, ...],
    ) -> None:
        self._require_columns(columns)

        for position in range(len(self.df)):
            missing = [
                _is_missing(self.df[column].iloc[position]) for column in columns
            ]

            if any(missing) and not all(missing):
                row = self.df.index[position]

                raise ValueError(
                    'columns must be all present or '
                    f'all absent at row {row!r}: '
                    f'{list(columns)}'
                )

    def _validate_literal(
        self,
        rule: _LiteralRule,
    ) -> None:
        self._require_columns((rule.column,))

        values = self.df[rule.column]

        for position in range(len(values)):
            value = values.iloc[position]

            if _is_missing(value):
                if rule.required:
                    row = self.df.index[position]

                    raise ValueError(
                        f'column {rule.column!r} is required at row {row!r}'
                    )

                continue

            valid = bool(pd.Series([value]).isin(rule.allowed).iloc[0])

            if not valid:
                row = self.df.index[position]

                raise ValueError(
                    f'invalid value {value!r} '
                    f'for {rule.column!r} '
                    f'at row {row!r}; '
                    f'allowed values are '
                    f'{list(rule.allowed)!r}'
                )

    def _require_columns(
        self,
        columns: collections.abc.Iterable[str],
    ) -> None:
        missing = set(columns).difference(self.df.columns)

        if missing:
            raise KeyError(f'unknown dataframe columns: {sorted(missing)}')

    # ------------------------------------------------------------------
    # metadata construction

    def _build_meta(
        self,
        name: str,
    ) -> pd.DataFrame:
        if self._keep_meta is not None:
            self._require_columns(self._keep_meta)

            meta = self.df.loc[
                :,
                list(self._keep_meta),
            ].copy()

        else:
            unknown = self._drop_meta.difference(self.df.columns)

            if unknown:
                raise KeyError(
                    f'cannot drop unknown metadata columns: {sorted(unknown)}'
                )

            columns = [
                column for column in self.df.columns if column not in self._drop_meta
            ]

            meta = self.df.loc[
                :,
                columns,
            ].copy()

        meta.reset_index(
            drop=True,
            inplace=True,
        )

        meta.index = pd.RangeIndex(
            len(meta),
            name=name,
        )

        return meta

    # ------------------------------------------------------------------
    # resource construction

    def _build_sources(
        self,
        name: str,
    ) -> SourcesGrouping:
        groups: list[Sources] = []
        entries: list[int] = []
        tags: list[Tag] = []

        for spec in self._source_specs:
            for position in range(len(self.df)):
                path_value = self._format_template(
                    spec.path,
                    position,
                    preserve_scalar=False,
                )

                if path_value is None:
                    if spec.required:
                        self._raise_missing_resource(
                            position,
                            category='source',
                            tag=spec.tag,
                        )

                    continue

                extra: dict[
                    str,
                    object,
                ] = {}

                for key, template in spec.extra.items():
                    value = self._format_template(
                        template,
                        position,
                        preserve_scalar=True,
                    )

                    if value is None:
                        row = self.df.index[position]

                        raise ValueError(
                            f'source {spec.tag!r} '
                            f'has path but missing '
                            f'extra value {key!r} '
                            f'at row {row!r}'
                        )

                    extra[key] = value

                path = self._resolve_path(
                    str(path_value),
                    kind=spec.kind,
                    exists=spec.exists,
                )

                source = Source(
                    path=path,
                    extra=extra,
                    kind=spec.kind,
                )

                groups.append(Sources.from_sources([source]))
                entries.append(position)
                tags.append(spec.tag)

        return SourcesGrouping.from_items(
            groups,
            meta=self._resource_meta(
                name,
                entries,
                tags,
            ),
        )

    def _build_windows(
        self,
        name: str,
    ) -> nocte.core.windows.WindowsGrouping:
        groups: list[nocte.core.windows.Windows] = []
        entries: list[int] = []
        tags: list[Tag] = []

        for spec in self._windows_specs:
            values = self.df[spec.column]

            for position in range(len(self.df)):
                raw = values.iloc[position]

                if _is_missing(raw):
                    if spec.required:
                        self._raise_missing_resource(
                            position,
                            category='windows',
                            tag=spec.tag,
                        )

                    continue

                windows = self._parse_windows(
                    raw,
                    spec=spec,
                    position=position,
                )

                if windows is None:
                    if spec.required:
                        self._raise_missing_resource(
                            position,
                            category='windows',
                            tag=spec.tag,
                        )

                    continue

                groups.append(windows)
                entries.append(position)
                tags.append(spec.tag)

        return nocte.core.windows.WindowsGrouping.from_items(
            groups,
            meta=self._resource_meta(
                name,
                entries,
                tags,
            ),
        )

    def _build_events(
        self,
        name: str,
    ) -> nocte.core.events.EventsGrouping:
        groups: list[nocte.core.events.Events] = []
        entries: list[int] = []
        tags: list[Tag] = []

        for spec in self._events_specs:
            values = self.df[spec.column]

            for position in range(len(self.df)):
                raw = values.iloc[position]

                if _is_missing(raw):
                    if spec.required:
                        self._raise_missing_resource(
                            position,
                            category='events',
                            tag=spec.tag,
                        )

                    continue

                events = self._parse_events(
                    raw,
                    spec=spec,
                    position=position,
                )

                if events is None:
                    if spec.required:
                        self._raise_missing_resource(
                            position,
                            category='events',
                            tag=spec.tag,
                        )

                    continue

                groups.append(events)
                entries.append(position)
                tags.append(spec.tag)

        return nocte.core.events.EventsGrouping.from_items(
            groups,
            meta=self._resource_meta(
                name,
                entries,
                tags,
            ),
        )

    def _parse_windows(
        self,
        value: object,
        *,
        spec: _WindowsSpec,
        position: int,
    ) -> nocte.core.windows.Windows | None:
        if spec.parser is None:
            if isinstance(
                value,
                nocte.core.windows.Windows,
            ):
                return value

            row = self.df.index[position]

            raise TypeError(
                f'windows resource {spec.tag!r} '
                f'at row {row!r} must contain '
                'Windows when no parser is supplied'
            )

        try:
            result = spec.parser(value)
        except Exception as exc:
            row = self.df.index[position]

            raise ValueError(
                f'failed to parse windows resource {spec.tag!r} at row {row!r}'
            ) from exc

        if result is not None and not isinstance(
            result,
            nocte.core.windows.Windows,
        ):
            raise TypeError(
                f'windows parser for {spec.tag!r} must return Windows or None'
            )

        return result

    def _parse_events(
        self,
        value: object,
        *,
        spec: _EventsSpec,
        position: int,
    ) -> nocte.core.events.Events | None:
        if spec.parser is None:
            if isinstance(
                value,
                nocte.core.events.Events,
            ):
                return value

            row = self.df.index[position]

            raise TypeError(
                f'events resource {spec.tag!r} '
                f'at row {row!r} must contain '
                'Events when no parser is supplied'
            )

        try:
            result = spec.parser(value)
        except Exception as exc:
            row = self.df.index[position]

            raise ValueError(
                f'failed to parse events resource {spec.tag!r} at row {row!r}'
            ) from exc

        if result is not None and not isinstance(
            result,
            nocte.core.events.Events,
        ):
            raise TypeError(
                f'events parser for {spec.tag!r} must return Events or None'
            )

        return result

    @staticmethod
    def _resource_meta(
        name: str,
        entries: list[int],
        tags: list[Tag],
    ) -> pd.DataFrame:
        if len(entries) != len(tags):
            raise ValueError('resource entries and tags must have equal length')

        meta = pd.DataFrame(
            {
                name: pd.Series(
                    entries,
                    dtype=np.int64,
                ),
                'tag': pd.Series(
                    tags,
                    dtype=object,
                ),
            }
        )

        meta.index = pd.RangeIndex(
            len(meta),
            name='group',
        )

        return meta

    def _raise_missing_resource(
        self,
        position: int,
        *,
        category: str,
        tag: Tag,
    ) -> typing.NoReturn:
        row = self.df.index[position]

        raise ValueError(
            f'required {category} resource {tag!r} is missing at row {row!r}'
        )

    # ------------------------------------------------------------------
    # templates

    @staticmethod
    def _template_fields(
        template: str,
    ) -> tuple[str, ...]:
        if not isinstance(
            template,
            str,
        ):
            raise TypeError('template must be a string')

        fields: list[str] = []

        for (
            _,
            field,
            _,
            _,
        ) in string.Formatter().parse(template):
            if field is None:
                continue

            if not field:
                raise ValueError('positional format fields are not supported')

            fields.append(field)

        return tuple(dict.fromkeys(fields))

    def _format_template(
        self,
        template: str,
        position: int,
        *,
        preserve_scalar: bool,
    ) -> object | None:
        parsed = list(string.Formatter().parse(template))

        fields = self._template_fields(template)

        values: dict[
            str,
            object,
        ] = {}

        for field in fields:
            if field not in self.df.columns:
                raise KeyError(f'unknown template column {field!r}')

            value = self.df[field].iloc[position]

            if _is_missing(value):
                return None

            values[field] = value

        # Preserve the native scalar type for a bare "{column}"
        # template. This is useful for Source.extra, where a sampling
        # rate should remain a float rather than become "30000.0".
        if preserve_scalar and len(parsed) == 1:
            (
                literal,
                field,
                format_spec,
                conversion,
            ) = parsed[0]

            if (
                literal == ''
                and field is not None
                and format_spec == ''
                and conversion is None
            ):
                return values[field]

        try:
            return template.format_map(values)
        except (
            KeyError,
            ValueError,
        ) as exc:
            row = self.df.index[position]

            raise ValueError(f'failed to format {template!r} at row {row!r}') from exc

    # ------------------------------------------------------------------
    # paths

    def _resolve_path(
        self,
        value: str,
        *,
        kind: SourceKind,
        exists: bool,
    ) -> pathlib.Path:
        candidates = self._path_candidates(value)

        matching = [
            candidate
            for candidate in candidates
            if self._path_matches(
                candidate,
                kind=kind,
            )
        ]

        if matching:
            return matching[0]

        existing = [candidate for candidate in candidates if candidate.exists()]

        if existing and kind != 'any':
            raise ValueError(f'path exists but is not of kind {kind!r}: {existing[0]}')

        if exists:
            raise FileNotFoundError(
                'resource path does not exist; '
                f'tried: '
                f'{[str(path) for path in candidates]}'
            )

        return candidates[0]

    def _path_candidates(
        self,
        value: str,
    ) -> list[pathlib.Path]:
        value = value.strip()

        if not value:
            raise ValueError('resource path cannot be empty')

        if not self._roots:
            return [pathlib.Path(value)]

        # A spreadsheet may contain a path beneath any one of several
        # equivalent roots, for example a Windows UNC mount while the
        # Builder is running on Linux. If it matches one configured root,
        # preserve the suffix and try it beneath every candidate root.
        relative: (
            tuple[
                str,
                ...,
            ]
            | None
        ) = None

        for root in self._roots:
            relative = self._relative_parts(
                value,
                root,
            )

            if relative is not None:
                break

        candidates: list[pathlib.Path] = []

        if relative is not None:
            for root in self._roots:
                candidates.append(pathlib.Path(root).joinpath(*relative))

        else:
            raw = pathlib.Path(value)

            if raw.is_absolute():
                candidates.append(raw)

            else:
                parts = self._portable_parts(value)

                for root in self._roots:
                    candidates.append(pathlib.Path(root).joinpath(*parts))

                candidates.append(raw)

        unique: list[pathlib.Path] = []
        seen: set[str] = set()

        for candidate in candidates:
            key = str(candidate)

            if key in seen:
                continue

            seen.add(key)
            unique.append(candidate)

        if not unique:
            raise ValueError('could not construct any candidate paths')

        return unique

    @staticmethod
    def _portable_parts(
        value: str,
    ) -> tuple[str, ...]:
        normalized = value.replace(
            '\\',
            '/',
        )

        return tuple(part for part in normalized.split('/') if part)

    @classmethod
    def _relative_parts(
        cls,
        value: str,
        root: str,
    ) -> tuple[str, ...] | None:
        value_parts = cls._portable_parts(value)
        root_parts = cls._portable_parts(root)

        if len(value_parts) < len(root_parts):
            return None

        prefix = tuple(part.casefold() for part in value_parts[: len(root_parts)])

        expected = tuple(part.casefold() for part in root_parts)

        if prefix != expected:
            return None

        return value_parts[len(root_parts) :]

    @staticmethod
    def _path_matches(
        path: pathlib.Path,
        *,
        kind: SourceKind,
    ) -> bool:
        if kind == 'file':
            return path.is_file()

        if kind == 'dir':
            return path.is_dir()

        return path.exists()

    # ------------------------------------------------------------------
    # declaration invariants

    @staticmethod
    def _require_new_tag(
        tag: Tag,
        *,
        specs: collections.abc.Sequence[_SourceSpec | _WindowsSpec | _EventsSpec],
        category: str,
    ) -> None:
        if any(spec.tag == tag for spec in specs):
            raise ValueError(f'duplicate {category} tag: {tag!r}')
