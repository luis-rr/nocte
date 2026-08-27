"""
Code to perform different dimensionality reductions, including supervisded and unsupervised.
Mostly wrappers around sklearn for nocte data classes.
"""

import numpy as np
import pandas as pd
import scipy.linalg

from nocte import traces as tr


def _take_pca(traces: pd.DataFrame, n_components: int | None = None):
    import sklearn.decomposition

    traces = traces.dropna()

    pca = sklearn.decomposition.PCA(n_components=n_components)

    valid = np.all(np.isfinite(traces), axis=1)

    transformed_array = pca.fit_transform(traces[valid].values)

    pc_idcs = np.arange(pca.n_components_) + 1

    transformed: pd.DataFrame = pd.DataFrame(
        transformed_array, columns=pc_idcs, index=traces.index
    )
    transformed = transformed.rename_axis(
        columns='PC',
        index=traces.index.name,
    )

    explained_variance = pd.Series(pca.explained_variance_, index=pc_idcs)
    explained_variance = explained_variance.rename('explained_variance')
    explained_variance.index.name = 'PC'

    components: pd.DataFrame = pd.DataFrame(
        pca.components_,
        columns=traces.columns,
        index=pc_idcs,
    )

    components: pd.DataFrame = components.rename_axis(
        index='PC',
        columns=traces.columns.name,
    )

    return transformed, explained_variance, components


def take_pca(traces: tr.Traces, n_components: int | None = None) -> tr.Traces:
    transformed, explained_variance, components = _take_pca(
        traces.traces,
        n_components=n_components,
    )

    reg = explained_variance.to_frame()
    reg = pd.concat(
        [
            reg,
            components.add_prefix('comp_'),
        ],
        axis=1,
    )

    return tr.Traces(
        reg=reg,
        traces=transformed,
    )


def _take_jpca(
    latent: pd.DataFrame,
    n_components: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute a jPCA-like projection from a latent trajectory.

    Parameters
    ----------
    latent:
        DataFrame of shape time x features. Index is assumed to be time.
    n_components:
        Number of output dimensions. If None, returns all available dimensions.
    return_info:
        If True, also return diagnostics and fitted components.

    Returns
    -------
    jpca:
        DataFrame with projected jPCA components.
    variance_explained:
        Series with variance captured by each jPCA axis.
    components:
        DataFrame with columns corresponding to original features and rows to jPCA axes.
    """
    X = latent.to_numpy(dtype=float)
    X = X - np.nanmean(X, axis=0, keepdims=True)

    if np.isnan(X).any():
        raise ValueError('latent contains NaNs. Interpolate/drop them before jPCA.')

    n_features = X.shape[1]

    if n_components is None:
        n_comp: int = n_features

    else:
        n_comp: int = n_components

    if n_comp > n_features:
        raise ValueError(
            f'n_components={n_comp} cannot exceed latent dimension {n_features}.'
        )

    time = latent.index.to_numpy(dtype=float)
    if len(time) > 1 and np.all(np.isfinite(time)) and np.all(np.diff(time) > 0):
        dX = np.gradient(X, time, axis=0)
    else:
        dX = np.gradient(X, axis=0)

    # Fit unconstrained linear dynamics: dX = X M
    M, *_ = np.linalg.lstsq(X, dX, rcond=None)

    dX_hat = X @ M
    ss_res = np.sum((dX - dX_hat) ** 2)
    ss_tot = np.sum((dX - dX.mean(axis=0, keepdims=True)) ** 2)
    dynamics_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Keep only rotational part.
    M_skew = 0.5 * (M - M.T)

    eigvals, eigvecs = np.linalg.eig(M_skew)

    # Sort by oscillatory strength.
    order = np.argsort(np.abs(np.imag(eigvals)))[::-1]

    basis = []
    mode_eigvals = []

    for idx in order:
        eigval = eigvals[idx]

        if np.abs(np.imag(eigval)) < 1e-12:
            continue

        v = eigvecs[:, idx]
        candidate = np.column_stack([np.real(v), np.imag(v)])

        # Avoid adding duplicate conjugate plane twice.
        candidate, _ = np.linalg.qr(candidate)

        for j in range(candidate.shape[1]):
            w = candidate[:, j]

            # Orthogonalize against already selected axes.
            for prev in basis:
                w = w - np.dot(prev, w) * prev

            norm = np.linalg.norm(w)
            if norm < 1e-12:
                continue

            basis.append(w / norm)
            mode_eigvals.append(eigval)

            if len(basis) >= n_comp:
                break

        if len(basis) >= n_comp:
            break

    if len(basis) < n_comp:
        raise RuntimeError(
            f'Only found {len(basis)} jPCA dimensions, but requested {n_comp}.'
        )

    W = np.column_stack(basis)
    Y = X @ W

    cols = [f'jpca_{i}' for i in range(n_comp)]
    transformed = pd.DataFrame(Y, index=latent.index, columns=cols)

    # Variance captured by each projected axis.
    total_var = np.sum(np.var(X, axis=0, ddof=1))
    axis_var = np.var(Y, axis=0, ddof=1)
    explained_variance = pd.Series(
        axis_var,
        index=cols,
        name='explained_variance',
    )

    components = pd.DataFrame(
        W.T,
        index=cols,
        columns=latent.columns,
    )

    return transformed, explained_variance, components


def take_jpca(traces: tr.Traces, n_components: int | None = None) -> tr.Traces:
    transformed, explained_variance, components = _take_jpca(
        traces.traces,
        n_components=n_components,
    )

    reg = explained_variance.to_frame()
    reg = pd.concat(
        [
            reg,
            components.add_prefix('comp_'),
        ],
        axis=1,
    )

    return tr.Traces(
        reg=reg,
        traces=transformed,
    )


def _take_tica(
    latent: pd.DataFrame,
    lag: float,
    n_components: int | None = None,
    regularization: float = 1e-6,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute a TICA projection from a latent trajectory.

    Parameters
    ----------
    latent:
        DataFrame of shape time x features. Index is assumed to be time.
    lag:
        Lag used for the time-lagged covariance.
        It is interpreted in the same units as the index.
    n_components:
        Number of output dimensions. If None, returns all available dimensions.
    regularization:
        Small diagonal regularizer added to the equal-time covariance matrix.

    Returns
    -------
    tica:
        DataFrame with projected TICA components.
    variance_explained:
        Series with variance captured by each TICA axis.
    components:
        DataFrame with columns corresponding to original features and rows to
        TICA axes.
    """
    X = latent.to_numpy(dtype=float)

    if np.isnan(X).any():
        raise ValueError('latent contains NaNs. Interpolate/drop them before TICA.')

    X = X - X.mean(axis=0, keepdims=True)

    n_samples, n_features = X.shape

    if n_components is None:
        n_comp: int = n_features
    else:
        n_comp: int = n_components

    if n_comp > n_features:
        raise ValueError(
            f'n_components={n_comp} cannot exceed latent dimension {n_features}.'
        )

    time = latent.index.to_numpy()

    assert (
        np.issubdtype(time.dtype, np.number)
        and len(time) > 1
        and np.all(np.isfinite(time.astype(float)))
        and np.all(np.diff(time.astype(float)) > 0)
    )

    dt = np.median(np.diff(time.astype(float)))
    lag_samples = int(round(float(lag) / dt))

    if lag_samples <= 0:
        raise ValueError('lag must correspond to at least one sample.')

    if lag_samples >= n_samples:
        raise ValueError(
            f'lag={lag} gives lag_samples={lag_samples}, which is too large '
            f'for n_samples={n_samples}.'
        )

    X0 = X[:-lag_samples]
    X1 = X[lag_samples:]

    # Equal-time covariance, symmetrized over both time-shifted views.
    C0 = (X0.T @ X0 + X1.T @ X1) / (2 * (len(X0) - 1))

    # Time-lagged covariance, symmetrized for standard reversible TICA.
    Ctau = (X0.T @ X1 + X1.T @ X0) / (2 * (len(X0) - 1))

    C0 = C0 + regularization * np.eye(n_features)

    # Solve Ctau w = lambda C0 w.
    eigvals, eigvecs = scipy.linalg.eigh(Ctau, C0)

    # Largest absolute eigenvalues are the strongest lag-predictive modes.
    order = np.argsort(np.abs(eigvals))[::-1]

    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    W = eigvecs[:, :n_comp]

    # Optional but useful: normalize projected components to avoid arbitrary scale.
    Y = X @ W

    cols = [f'tica_{i}' for i in range(n_comp)]

    tica = pd.DataFrame(
        Y,
        index=latent.index,
        columns=cols,
    )

    total_var = np.sum(np.var(X, axis=0, ddof=1))
    axis_var = np.var(Y, axis=0, ddof=1)

    variance_explained = pd.Series(
        axis_var / total_var if total_var > 0 else np.nan,
        index=cols,
    )

    components = pd.DataFrame(
        W.T,
        index=cols,
        columns=latent.columns,
    )

    return tica, variance_explained, components


def take_tica(
    traces: tr.Traces,
    lag: float,
    n_components: int | None = None,
) -> tr.Traces:
    transformed, explained_variance, components = _take_tica(
        traces.traces,
        lag=lag,
        n_components=n_components,
    )

    reg = explained_variance.to_frame()
    reg = pd.concat(
        [
            reg,
            components.add_prefix('comp_'),
        ],
        axis=1,
    )

    return tr.Traces(
        reg=reg,
        traces=transformed,
    )


def _take_jpca_lagged(
    latent: pd.DataFrame,
    lag: float,
    n_components: int | None = None,
    regularization: float = 1e-6,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute a jPCA-like projection from a latent trajectory.

    This estimates a lagged linear dynamics operator:

        X(t + lag) ≈ X(t) A

    then extracts the strongest rotational planes from the skew-symmetric
    component of A.

    Parameters
    ----------
    latent:
        DataFrame of shape time x features. Index is assumed to be time.
    lag:
        Lag in the same units as the index.
    n_components:
        Number of output dimensions. If None, returns all available dimensions.
    regularization:
        Ridge regularization for the lagged linear regression.

    Returns
    -------
    jPCA:
        DataFrame with projected jPCA components.
    variance_explained:
        Series with variance captured by each jPCA axis.
    components:
        DataFrame with columns corresponding to original features and rows to
        jPCA axes.
    """
    X = latent.to_numpy(dtype=float)

    if np.isnan(X).any():
        raise ValueError('latent contains NaNs. Interpolate/drop them before jPCA.')

    X = X - X.mean(axis=0, keepdims=True)

    n_samples, n_features = X.shape

    if n_components is None:
        n_comp: int = n_features
    else:
        n_comp: int = n_components

    if n_comp > n_features:
        raise ValueError(
            f'n_components={n_comp} cannot exceed latent dimension {n_features}.'
        )

    time = latent.index.to_numpy()

    assert (
        np.issubdtype(time.dtype, np.number)
        and len(time) > 1
        and np.all(np.isfinite(time.astype(float)))
        and np.all(np.diff(time.astype(float)) > 0)
    )

    dt = np.median(np.diff(time.astype(float)))
    lag_samples = int(round(float(lag) / dt))

    if lag_samples <= 0:
        raise ValueError('lag must correspond to at least one sample.')

    if lag_samples >= n_samples:
        raise ValueError(
            f'lag={lag} gives lag_samples={lag_samples}, which is too large '
            f'for n_samples={n_samples}.'
        )

    X0 = X[:-lag_samples]
    X1 = X[lag_samples:]

    # Estimate lagged linear map X1 ≈ X0 A with ridge regularization.
    # A = (X0.T X0 + alpha I)^-1 X0.T X1
    C00 = X0.T @ X0
    C01 = X0.T @ X1

    A = np.linalg.solve(
        C00 + regularization * np.eye(n_features),
        C01,
    )

    # Extract rotational / antisymmetric component.
    A_skew = 0.5 * (A - A.T)

    eigvals, eigvecs = np.linalg.eig(A_skew)

    # Sort by rotational strength.
    order = np.argsort(np.abs(np.imag(eigvals)))[::-1]

    basis = []

    for idx in order:
        eigval = eigvals[idx]

        if np.abs(np.imag(eigval)) < 1e-12:
            continue

        v = eigvecs[:, idx]
        candidate = np.column_stack([np.real(v), np.imag(v)])

        candidate, _ = np.linalg.qr(candidate)

        for j in range(candidate.shape[1]):
            w = candidate[:, j]

            # Orthogonalize against already selected axes.
            for prev in basis:
                w = w - np.dot(prev, w) * prev

            norm = np.linalg.norm(w)
            if norm < 1e-12:
                continue

            basis.append(w / norm)

            if len(basis) >= n_comp:
                break

        if len(basis) >= n_comp:
            break

    if len(basis) < n_comp:
        raise RuntimeError(
            f'Only found {len(basis)} jPCA dimensions, but requested {n_comp}.'
        )

    W = np.column_stack(basis)
    Y = X @ W

    cols = [f'jpca_{i}' for i in range(n_comp)]

    jpca = pd.DataFrame(
        Y,
        index=latent.index,
        columns=cols,
    )

    total_var = np.sum(np.var(X, axis=0, ddof=1))
    axis_var = np.var(Y, axis=0, ddof=1)

    variance_explained = pd.Series(
        axis_var / total_var if total_var > 0 else np.nan,
        index=cols,
    )

    components = pd.DataFrame(
        W.T,
        index=cols,
        columns=latent.columns,
    )

    return jpca, variance_explained, components


def take_jpca_lagged(
    traces: tr.Traces,
    lag: float,
    n_components: int | None = None,
) -> tr.Traces:
    transformed, explained_variance, components = _take_jpca_lagged(
        traces.traces,
        lag=lag,
        n_components=n_components,
    )

    reg = explained_variance.to_frame()
    reg = pd.concat(
        [
            reg,
            components.add_prefix('comp_'),
        ],
        axis=1,
    )

    return tr.Traces(
        reg=reg,
        traces=transformed,
    )


def _take_cca_lagged(
    latent: pd.DataFrame,
    lag: float,
    n_components: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Compute a lagged-CCA projection from a latent trajectory.

    Parameters
    ----------
    latent:
        DataFrame of shape time x features. Index is assumed to be time in
        floating point.
    lag:
        Lag used for the time-lagged covariance.
        It is interpreted in the same units as the index.
    n_components:
        Number of output dimensions. If None, returns all available dimensions.

    Returns
    -------
    transformed:
        DataFrame with projected CCA components.
    variance_explained:
        Series with variance captured by each CCA axis.
        Note: unlike PCA, these values are not necessarily additive because CCA
        axes are not generally orthogonal.
    components:
        DataFrame with columns corresponding to original features and rows to
        CCA axes.
    """
    if latent.empty:
        raise ValueError('latent is empty.')

    if not latent.index.is_monotonic_increasing:
        raise ValueError('latent index must be monotonically increasing.')

    if lag <= 0:
        raise ValueError('lag must be positive for a t -> t + lag projection.')

    times = latent.index.to_numpy(dtype=float)
    values = latent.to_numpy(dtype=float)

    if np.any(~np.isfinite(values)):
        raise ValueError('latent contains non-finite values.')

    n_features = values.shape[1]

    if n_components is None:
        n_components = n_features

    if not 1 <= n_components <= n_features:
        raise ValueError(
            f'n_components must be between 1 and {n_features}, got {n_components}.'
        )

    target_times = times + lag
    valid = target_times <= times[-1]

    if valid.sum() <= n_features:
        raise ValueError(
            'Not enough valid time-lagged samples. Try a smaller lag or a longer trajectory.'
        )

    x = values[valid]
    y = np.column_stack(
        [np.interp(target_times[valid], times, values[:, i]) for i in range(n_features)]
    )

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)

    x_centered = x - x_mean
    y_centered = y - y_mean

    denom = len(x_centered) - 1
    c_xx = x_centered.T @ x_centered / denom
    c_yy = y_centered.T @ y_centered / denom
    c_xy = x_centered.T @ y_centered / denom

    def _inverse_sqrt_psd(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(cov)

        scale = max(eigvals.max(), eps)
        keep = eigvals > eps * scale

        if not np.any(keep):
            raise ValueError('Covariance matrix is numerically rank deficient.')

        return (eigvecs[:, keep] / np.sqrt(eigvals[keep])) @ eigvecs[:, keep].T

    c_xx_inv_sqrt = _inverse_sqrt_psd(c_xx)
    c_yy_inv_sqrt = _inverse_sqrt_psd(c_yy)

    # Whitened lagged covariance. Its singular vectors define the CCA weights.
    m = c_xx_inv_sqrt @ c_xy @ c_yy_inv_sqrt
    u, _, _ = np.linalg.svd(m, full_matrices=False)

    # Left/present-time CCA weights: directions in X_t.
    weights = c_xx_inv_sqrt @ u[:, :n_components]

    # Normalize weights in ordinary Euclidean space so projected variances are
    # at least on a comparable scale. This does not make them orthogonal.
    weights /= np.linalg.norm(weights, axis=0, keepdims=True)

    # Fix arbitrary sign for visual stability.
    for i in range(weights.shape[1]):
        j = np.argmax(np.abs(weights[:, i]))
        if weights[j, i] < 0:
            weights[:, i] *= -1

    names = [f'cca_lagged_{i}' for i in range(n_components)]

    transformed_values = (values - x_mean) @ weights

    transformed = pd.DataFrame(
        transformed_values,
        index=latent.index,
        columns=names,
    )

    components = pd.DataFrame(
        weights.T,
        index=names,
        columns=latent.columns,
    )

    total_variance = np.var(values - values.mean(axis=0), axis=0, ddof=1).sum()
    axis_variance = np.var(transformed_values, axis=0, ddof=1)

    variance_explained = pd.Series(
        axis_variance / total_variance,
        index=names,
        name='variance_explained',
    )

    return transformed, variance_explained, components


def take_cca_lagged(
    traces: tr.Traces,
    lag: float,
    n_components: int | None = None,
) -> tr.Traces:
    transformed, explained_variance, components = _take_cca_lagged(
        traces.traces,
        lag=lag,
        n_components=n_components,
    )

    reg = explained_variance.to_frame()
    reg = pd.concat(
        [
            reg,
            components.add_prefix('comp_'),
        ],
        axis=1,
    )

    return tr.Traces(
        reg=reg,
        traces=transformed,
    )


def _reduced_rank_regression(
    neural_data: np.ndarray, behavior_data: np.ndarray, n_latents: int
):
    """
    Reduced-rank regression from neural activity to behavior.

    Parameters
    ----------
    neural_data : array, shape (T, N)
        Neural activity over time
    behavior_data : array, shape (T, B)
        Behavioral variables over time
    n_latents : int
        Number of latent dimensions

    Returns
    -------
    latent_activity : array, shape (T, n_latents)
        Low-dimensional neural activity
    neural_projection : array, shape (N, n_latents)
        Neural-to-latent projection matrix
    behavior_readout : array, shape (n_latents, B)
        Latent-to-behavior readout matrix
    behavior_prediction : array, shape (T, B)
        Predicted behavior
    """
    # Ordinary least squares mapping from neural activity to behavior
    full_regression_weights = np.linalg.lstsq(neural_data, behavior_data, rcond=None)[
        0
    ]  # (N, B)

    # Behavior predicted by the full model
    full_behavior_prediction = neural_data @ full_regression_weights

    # SVD of predicted behavior to find dominant behavioral subspace
    left_singular_vectors, singular_values, right_singular_vectors_t = scipy.linalg.svd(
        full_behavior_prediction, full_matrices=False
    )

    # Top behavioral directions
    behavior_subspace = right_singular_vectors_t[:n_latents].T  # (B, n_latents)

    # Reduced-rank mappings
    neural_projection = full_regression_weights @ behavior_subspace  # (N, n_latents)
    behavior_readout = behavior_subspace.T  # (n_latents, B)

    # Latent neural activity
    latent = neural_data @ neural_projection

    # Final behavioral prediction
    behavior_prediction = latent @ behavior_readout

    return latent, neural_projection, behavior_readout, behavior_prediction


def _reduced_rank_regression_dd(
    neural_data: pd.DataFrame, behavior_data: pd.DataFrame, n_latents: int
):
    valid = np.all(np.isfinite(neural_data.values), axis=1) & np.all(
        np.isfinite(neural_data.values), axis=1
    )

    latent_array, neural_projection, behavior_readout, behavior_prediction = (
        _reduced_rank_regression(
            neural_data.values[valid],
            behavior_data.values[valid],
            n_latents=n_latents,
        )
    )

    latent_idcs = np.arange(n_latents) + 1

    latent = pd.DataFrame(
        latent_array,
        index=neural_data.index[valid],
        columns=latent_idcs,
    )

    latent = latent.rename_axis(
        columns='PC',
        index=neural_data.index.name,
    )

    return latent, neural_projection, behavior_readout, behavior_prediction
