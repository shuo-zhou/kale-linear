import numpy as np
from numpy.linalg import inv, multi_dot
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph


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
    n = X.shape[0]
    knn_graph = kneighbors_graph(X, n_neighbour, metric=metric, mode=mode).toarray()
    W = np.zeros((n, n))
    knn_idx = np.logical_or(knn_graph, knn_graph.T)
    if mode == "distance":
        graph_kernel = pairwise_distances(X, metric=metric)
        W[knn_idx] = graph_kernel[knn_idx]
    else:
        W[knn_idx] = 1

    D = np.diag(np.sum(W, axis=1))
    if normalise:
        D_ = inv(sqrtm(D))
        lap_mat = np.eye(n) - multi_dot([D_, W, D_])
    else:
        lap_mat = D - W
    return lap_mat


def mmd_coef(ns, nt, ys=None, yt=None, kind="marginal", mu=0.5):
    n = ns + nt
    e = np.zeros((n, 1))
    e[:ns, 0] = 1.0 / ns
    e[ns:, 0] = -1.0 / nt
    M = np.dot(e, e.T)  # marginal mmd coefficients

    if kind == "joint" and ys is not None:
        Mc = 0  # conditional mmd coefficients
        class_all = np.unique(ys)
        if yt is not None and class_all.all() != np.unique(yt).all():
            raise ValueError("Source and target domain should have the same labels")

        for c in class_all:
            es = np.zeros([ns, 1])
            es[np.where(ys == c)] = 1.0 / (np.where(ys == c)[0].shape[0])
            et = np.zeros([nt, 1])
            if yt is not None:
                et[np.where(yt == c)[0]] = -1.0 / np.where(yt == c)[0].shape[0]
            e = np.vstack((es, et))
            e[np.where(np.isinf(e))[0]] = 0
            Mc = Mc + mu * np.dot(e, e.T)
        M = (1 - mu) * M + mu * Mc  # joint mmd coefficients
    return M


def base_init(X, kernel="linear", **kwargs):
    n = X.shape[0]
    # Construct kernel matrix
    ker_x = pairwise_kernels(X, metric=kernel, filter_params=True, **kwargs)
    ker_x[np.isnan(ker_x)] = 0

    unit_mat = np.eye(n)
    # Construct centering matrix
    ctr_mat = unit_mat - 1.0 / n * np.ones((n, n))

    return ker_x, unit_mat, ctr_mat, n
