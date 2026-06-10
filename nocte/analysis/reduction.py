"""
Code to perform different dimensionality reductions, including supervisded and unsupervised.
Mostly wrappers around sklearn for nocte data classes.
"""

import pandas as pd
import numpy as np
import scipy.linalg


from nocte import traces as tr

def _take_pca(traces, n_components=None):
    import sklearn.decomposition

    traces = traces.dropna()

    pca = sklearn.decomposition.PCA(n_components=n_components)

    valid = np.all(np.isfinite(traces), axis=1)

    transformed_array = pca.fit_transform(
        traces[valid].values
    )

    pc_idcs = np.arange(pca.n_components_) + 1

    transformed: pd.DataFrame = pd.DataFrame(transformed_array, columns=pc_idcs, index=traces.index)
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


def take_pca(traces: tr.Traces, n_components: int = None) -> tr.Traces:
    transformed, explained_variance, components = _take_pca(
        traces.traces,
        n_components=n_components,
    )

    reg = explained_variance.to_frame()
    reg = pd.concat([
        reg,
        components.add_prefix('comp_'),
    ], axis=1)

    return tr.Traces(
        reg=reg,
        traces=transformed,
    )


def _reduced_rank_regression(neural_data: np.ndarray, behavior_data: np.ndarray, n_latents: int):
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
    full_regression_weights = np.linalg.lstsq(
        neural_data, behavior_data, rcond=None
    )[0]  # (N, B)

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
    behavior_readout = behavior_subspace.T                           # (n_latents, B)

    # Latent neural activity
    latent_activity = neural_data @ neural_projection

    # Final behavioral prediction
    behavior_prediction = latent_activity @ behavior_readout

    return latent_activity, neural_projection, behavior_readout, behavior_prediction
