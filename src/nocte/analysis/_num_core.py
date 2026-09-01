"""Shared private helpers for sampled-data analysis."""

from __future__ import annotations

import collections.abc
import typing

import numpy as np
import numpy.typing as npt
import pandas as pd

import nocte.core.sampling
import nocte.core.traces

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
IntArray = npt.NDArray[np.intp]


class Bounds(typing.NamedTuple):
    """Finite half-open sample bounds for traces."""

    first: IntArray
    stop: IntArray

    @classmethod
    def from_traces(
        cls,
        traces: nocte.core.traces.Traces,
        positions: np.ndarray | None = None,
        *,
        desc: str = 'traces',
    ) -> typing.Self:
        """Validate trace support and return it in kernel-friendly form."""
        first, stop = traces.valid_bounds(
            positions,
            desc=desc,
        )

        return cls(
            first=np.ascontiguousarray(first, dtype=np.intp),
            stop=np.ascontiguousarray(stop, dtype=np.intp),
        )

    def are_full(
        self,
        n_samples: int,
    ) -> bool:
        """Return whether every bound spans the complete source grid."""
        return bool(np.all(self.first == 0) and np.all(self.stop == n_samples))


def duration_samples(
    sampling: nocte.core.sampling.SamplingRate,
    duration: float,
    *,
    desc: str,
) -> int:
    """Convert a positive duration to the nearest whole number of samples."""
    duration = float(duration)

    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f'{desc} must be finite and positive')

    duration = sampling.round_to_period(
        duration,
        desc=desc,
    )

    samples = sampling.ms_to_samples_exact(
        duration,
        desc=desc,
    )

    if samples <= 0:
        raise ValueError(f'{desc} must span at least one sample')

    return samples


def to_db(
    values: FloatArray,
) -> FloatArray:
    """Convert power-like values to decibels."""
    with np.errstate(
        divide='ignore',
        invalid='ignore',
    ):
        result = 10.0 * np.log10(values)

    return np.asarray(
        result,
        dtype=np.float64,
    )


def traces_like(
    source: nocte.core.traces.Traces,
    values: np.ndarray,
    *,
    meta: pd.DataFrame | None = None,
) -> nocte.core.traces.Traces:
    """Build derived traces on the exact source grid without dtype coercion."""
    return nocte.core.traces.Traces.from_grid(
        values,
        source.grid,
        meta=(source.meta if meta is None else meta),
    )


def provenance_name(
    source_name: str,
    *,
    reserved: collections.abc.Collection[str] = (),
) -> str:
    """Return a source-identity column name that avoids structural collisions."""
    if source_name in reserved:
        return f'source_{source_name}'

    return source_name


def set_provenance(
    meta: pd.DataFrame,
    name: str,
    values: np.ndarray,
) -> None:
    """Set provenance without silently contradicting existing metadata."""
    if name not in meta.columns:
        meta[name] = values
        return

    if not np.array_equal(
        meta[name].to_numpy(copy=False),
        values,
    ):
        raise ValueError(f'metadata column {name!r} contradicts analysis provenance')


def feature_meta(
    source_meta: pd.DataFrame,
    *,
    source_ids: np.ndarray,
    source_name: str,
    feature_name: str,
    features: collections.abc.Sequence[typing.Any] | np.ndarray,
    result_name: str,
) -> pd.DataFrame:
    """Expand source metadata over one derived feature axis."""
    source_ids = np.asarray(source_ids)

    if source_ids.ndim != 1:
        raise ValueError('source_ids must be one-dimensional')

    if len(source_meta) != len(source_ids):
        raise ValueError('source metadata and source_ids must have equal length')

    feature_values = np.asarray(features)

    if feature_values.ndim != 1:
        raise ValueError('features must be one-dimensional')

    n_features = len(feature_values)

    take = np.repeat(
        np.arange(
            len(source_meta),
            dtype=np.intp,
        ),
        n_features,
    )

    meta = source_meta.iloc[take].copy()

    prov = provenance_name(
        source_name,
        reserved=(
            result_name,
            feature_name,
        ),
    )

    set_provenance(
        meta,
        prov,
        np.repeat(
            source_ids,
            n_features,
        ),
    )

    if feature_name in meta.columns:
        raise ValueError(
            f'metadata column {feature_name!r} conflicts with derived feature metadata'
        )

    meta[feature_name] = np.tile(
        feature_values,
        len(source_meta),
    )

    meta.index = pd.RangeIndex(
        len(meta),
        name=result_name,
    )

    return meta


def feature_traces(
    values: np.ndarray,
    grid: nocte.core.sampling.TimeGrid,
    *,
    source_meta: pd.DataFrame,
    source_ids: np.ndarray,
    source_name: str,
    feature_name: str,
    features: collections.abc.Sequence[typing.Any] | np.ndarray,
    result_name: str,
) -> nocte.core.traces.Traces:
    """
    Build traces from a `(source, feature, time)` analysis result.

    Source metadata is repeated over features, source identity is retained as
    provenance, the feature becomes item metadata, and `(source, feature)` is
    flattened into the result item axis.
    """
    values = np.asarray(values)

    if values.ndim != 3:
        raise ValueError('values must have shape (source, feature, time)')

    feature_values = np.asarray(features)

    if feature_values.ndim != 1:
        raise ValueError('features must be one-dimensional')

    if values.shape[0] != len(source_meta):
        raise ValueError('values source axis does not match source metadata')

    if values.shape[1] != len(feature_values):
        raise ValueError('values feature axis does not match features')

    if values.shape[2] != grid.n_samples:
        raise ValueError('values time axis does not match output grid')

    meta = feature_meta(
        source_meta,
        source_ids=source_ids,
        source_name=source_name,
        feature_name=feature_name,
        features=feature_values,
        result_name=result_name,
    )

    flattened = values.reshape(
        values.shape[0] * values.shape[1],
        values.shape[2],
    )

    return nocte.core.traces.Traces.from_grid(
        flattened,
        grid,
        meta=meta,
    )
