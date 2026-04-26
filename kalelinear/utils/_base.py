import numpy as np
from numpy.linalg import inv, multi_dot
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph

from ._backend import infer_backend, to_backend, to_numpy


def lap_norm(X, n_neighbour=3, metric="cosine", mode="distance", normalise=True):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    n_neighbour : int, optional
        [description], by default 3
    metric : str, optional
        [description], by default 'cosine'
    mode : str, optional
        {‘connectivity’, ‘distance’}, by default 'distance'. Type of
        returned matrix: ‘connectivity’ will return the connectivity
        matrix with ones and zeros, and ‘distance’ will return the
        distances between neighbors according to the given metric.
    normalise : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    backend = infer_backend(X)
    x_np = to_numpy(X)
    n = x_np.shape[0]
    knn_graph = kneighbors_graph(x_np, n_neighbour, metric=metric, mode=mode).toarray()
    W = np.zeros((n, n))
    knn_idx = np.logical_or(knn_graph, knn_graph.T)
    if mode == "distance":
        graph_kernel = pairwise_distances(x_np, metric=metric)
        W[knn_idx] = graph_kernel[knn_idx]
    else:
        W[knn_idx] = 1

    D = np.diag(np.sum(W, axis=1))
    if normalise:
        D_ = inv(sqrtm(D))
        lap_mat = np.eye(n) - multi_dot([D_, W, D_])
    else:
        lap_mat = D - W
    return to_backend(lap_mat, backend, reference=X)


def mmd_coef(ns, nt, ys=None, yt=None, kind="marginal", mu=0.5):
    backend = infer_backend(ys, yt)
    ys_np = to_numpy(ys) if ys is not None else None
    yt_np = to_numpy(yt) if yt is not None else None
    n = ns + nt
    e = np.zeros((n, 1))
    e[:ns, 0] = 1.0 / ns
    e[ns:, 0] = -1.0 / nt
    M = np.dot(e, e.T)  # marginal mmd coefficients

    if kind == "joint" and ys_np is not None:
        Mc = 0  # conditional mmd coefficients
        class_all = np.unique(ys_np)
        if yt_np is not None:
            target_classes = np.unique(yt_np)
            if not np.array_equal(class_all, target_classes):
                raise ValueError("Source and target domain should have the same labels")

        for c in class_all:
            es = np.zeros([ns, 1])
            es[np.where(ys_np == c)] = 1.0 / (np.where(ys_np == c)[0].shape[0])
            et = np.zeros([nt, 1])
            if yt_np is not None:
                et[np.where(yt_np == c)[0]] = -1.0 / np.where(yt_np == c)[0].shape[0]
            e = np.vstack((es, et))
            e[np.where(np.isinf(e))[0]] = 0
            Mc = Mc + np.dot(e, e.T)
        M = (1 - mu) * M + mu * Mc  # joint mmd coefficients
    return to_backend(M, backend, reference=ys if ys is not None else yt)


def centering_matrix(size, dtype=np.float64):
    """Generate a centering matrix."""
    unit_matrix = np.eye(size, dtype=dtype)
    return unit_matrix - 1.0 / size * np.ones((size, size), dtype=dtype)


def kernel_fit_matrices(X, kernel="linear", metric=None, filter_params=True, return_backend=False, **kwargs):
    """Prepare common fit-time kernel, identity, and centering matrices."""
    backend = infer_backend(X)
    x_np = to_numpy(X)
    n = x_np.shape[0]
    kernel_metric = kernel if metric is None else metric

    x_kernel_matrix = pairwise_kernels(x_np, metric=kernel_metric, filter_params=filter_params, **kwargs)
    x_kernel_matrix[np.isnan(x_kernel_matrix)] = 0

    unit_matrix = np.eye(n)
    h_matrix = centering_matrix(n)

    if not return_backend:
        return x_kernel_matrix, unit_matrix, h_matrix, n

    return (
        to_backend(x_kernel_matrix, backend, reference=X),
        to_backend(unit_matrix, backend, reference=X),
        to_backend(h_matrix, backend, reference=X),
        n,
    )


def base_init(X, kernel="linear", **kwargs):
    return kernel_fit_matrices(X, kernel=kernel, return_backend=True, **kwargs)
