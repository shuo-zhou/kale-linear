import warnings

import numpy as np
import osqp
import scipy.sparse as sparse
from cvxopt import matrix, solvers
from numpy.linalg import inv, multi_dot
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer

from kalelinear._domain import split_source_target
from kalelinear.utils import infer_backend, kernel_fit_matrices, to_backend, to_numpy


class BaseKaleEstimator(BaseEstimator, ClassifierMixin):
    """Base class for Kale-Linear classifiers."""

    def __init__(
        self,
        kernel="linear",
        k_neighbour=5,
        manifold_metric="cosine",
        knn_mode="distance",
        pos_label=1,
        neg_label=-1,
        **kwargs,
    ):
        super().__init__()
        self.kernel = kernel
        self.k_neighbour = k_neighbour
        self.manifold_metric = manifold_metric
        self.knn_mode = knn_mode
        self.coef_ = None
        self.support_ = None
        self._lb = LabelBinarizer(pos_label=pos_label, neg_label=neg_label)
        self.kwargs = kwargs
        self.backend_ = "numpy"
        self.X = None
        self.y = None

    @classmethod
    def _solve_semi_dual(cls, x_kernel_matrix, y, obj_core, C, solver="osqp"):
        x_kernel_matrix = to_numpy(x_kernel_matrix)
        y = to_numpy(y)
        obj_core = to_numpy(obj_core)
        if len(y.shape) == 1:
            coef_, support_ = cls._semi_binary_dual(x_kernel_matrix, y, obj_core, C, solver)
            support_ = [support_]
        else:
            coef_ = np.zeros((x_kernel_matrix.shape[1], y.shape[1]))
            support_ = []
            for i in range(y.shape[1]):
                coef_i, support_i = cls._semi_binary_dual(x_kernel_matrix, y[:, i], obj_core, C, solver)
                coef_[:, i] = coef_i
                support_.append(support_i)

        return coef_, support_

    @classmethod
    def _semi_binary_dual(cls, x_kernel_matrix, y, obj_core, C, solver="osqp"):
        x_kernel_matrix = to_numpy(x_kernel_matrix)
        y = to_numpy(y)
        obj_core = to_numpy(obj_core)
        n_labeled = y.shape[0]
        n = x_kernel_matrix.shape[0]
        J = np.zeros((n_labeled, n))
        J[:n_labeled, :n_labeled] = np.eye(n_labeled)
        obj_inv = inv(obj_core)
        y_diag = np.diag(y.reshape(-1))
        Q = multi_dot([y_diag, J, x_kernel_matrix, obj_inv, J.T, y_diag]).astype("float32")
        alpha = cls._quadprog(Q, y, C, solver)
        coef_ = multi_dot([obj_inv, J.T, y_diag, alpha])
        support_ = np.flatnonzero(alpha > 1e-8)
        return coef_, support_

    @classmethod
    def _quadprog(cls, P, y, C, solver="osqp"):
        P = to_numpy(P)
        y = to_numpy(y)
        n_labeled = y.shape[0]
        q = -1 * np.ones(n_labeled)
        upper_bound = C / n_labeled

        if solver == "cvxopt":
            G = np.zeros((2 * n_labeled, n_labeled))
            G[:n_labeled, :] = -1 * np.eye(n_labeled)
            G[n_labeled:, :] = np.eye(n_labeled)
            h = np.zeros((2 * n_labeled, 1))
            h[n_labeled:, :] = upper_bound

            P = matrix(P.astype(np.float64))
            q = matrix(q.reshape(-1, 1))
            G = matrix(G)
            h = matrix(h)
            A = matrix(y.reshape(1, -1).astype(np.float64))
            b = matrix(np.zeros(1).astype(np.float64))

            solvers.options["show_progress"] = False
            sol = solvers.qp(P, q, G, h, A, b)
            beta = np.array(sol["x"]).reshape(n_labeled)
        elif solver == "osqp":
            warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
            P_sparse = sparse.csc_matrix(P)
            G = sparse.vstack([sparse.eye(n_labeled), y.reshape(1, -1)]).tocsc()
            l_ = np.zeros(n_labeled + 1)
            u = np.zeros(n_labeled + 1)
            u[:n_labeled] = upper_bound

            prob = osqp.OSQP()
            prob.setup(P_sparse, q, G, l_, u, verbose=False)
            beta = prob.solve().x
        else:
            raise ValueError("Invalid QP solver")

        return beta

    @classmethod
    def _solve_semi_ls(cls, Q, y):
        Q = to_numpy(Q)
        y = to_numpy(y)
        n = Q.shape[0]
        n_labeled = y.shape[0]
        Q_inv = inv(Q)
        if len(y.shape) == 1:
            y_ = np.zeros(n)
            y_[:n_labeled] = y
        else:
            y_ = np.zeros((n, y.shape[1]))
            y_[:n_labeled, :] = y
        return np.dot(Q_inv, y_)

    def _get_fit_data(self):
        if getattr(self, "X", None) is not None:
            return to_numpy(self.X)
        raise NotFittedError("This estimator is not fitted yet. Call 'fit' before using this estimator.")

    def decision_function(self, X):
        backend = infer_backend(X)
        x_np = to_numpy(X)
        x_kernel_matrix = pairwise_kernels(
            x_np, self._get_fit_data(), metric=self.kernel, filter_params=True, **self.kwargs
        )
        scores = np.dot(x_kernel_matrix, to_numpy(self.coef_))
        return to_backend(scores, backend, reference=X)

    def predict(self, X):
        backend = infer_backend(X)
        scores = to_numpy(self.decision_function(X))
        y_pred = self._lb.inverse_transform(scores)
        return to_backend(y_pred, backend, reference=X)


class BaseDomainAdaptationEstimator(BaseKaleEstimator):
    """Base class for domain adaptation estimators."""

    @staticmethod
    def _prepare_kernel_fit_data(X, y=None, covariates=None, kernel="linear", **kwargs):
        """Prepare numpy fit arrays and common kernel matrices."""
        metric = kwargs.pop("metric", kernel)
        filter_params = kwargs.pop("filter_params", True)
        X = to_numpy(X)
        y = to_numpy(y)
        covariates = to_numpy(covariates)
        x_kernel_matrix, unit_matrix, centering_matrix, n = kernel_fit_matrices(
            X,
            metric=metric,
            filter_params=filter_params,
            **kwargs,
        )

        return X, y, covariates, x_kernel_matrix, unit_matrix, centering_matrix, n

    def _split_source_target_by_covariate(
        self,
        X,
        y,
        covariates,
        target_covariate=None,
        unlabeled_value=None,
    ):
        """Split combined domain data into source and target blocks."""
        if covariates is None:
            raise ValueError("`covariates` must be provided to identify source and target domains.")

        X = to_numpy(X)
        y = to_numpy(y)
        covariates = to_numpy(covariates)

        if y is None:
            raise ValueError("`y` must contain source labels.")
        split = split_source_target(X, covariates, target_covariate=target_covariate)
        source_idx = split.source_idx
        target_idx = split.target_idx
        ns = source_idx.shape[0]
        n_samples = X.shape[0]

        y = np.asarray(y)
        if y.shape[0] == ns:
            ys = y
            target_labeled_idx = np.array([], dtype=int)
        elif y.shape[0] == n_samples:
            ys = y[source_idx]
            target_labels = y[target_idx]
            if unlabeled_value is None:
                target_labeled_mask = np.ones(target_idx.shape[0], dtype=bool)
            else:
                target_labeled_mask = target_labels != unlabeled_value
            target_labeled_idx = target_idx[target_labeled_mask]
        else:
            raise ValueError("`y` must contain either source labels only or one label per sample in `X`.")

        target_unlabeled_idx = np.setdiff1d(target_idx, target_labeled_idx, assume_unique=True)
        target_fit_idx = np.concatenate([target_labeled_idx, target_unlabeled_idx])
        yt = y[target_labeled_idx] if target_labeled_idx.size > 0 else None

        return {
            "Xs": X[source_idx],
            "ys": ys,
            "Xt": X[target_fit_idx],
            "yt": yt,
            "source_idx": source_idx,
            "target_idx": target_idx,
            "target_fit_idx": target_fit_idx,
            "target_covariate": split.target_covariate,
        }


__all__ = ["BaseKaleEstimator", "BaseDomainAdaptationEstimator"]
