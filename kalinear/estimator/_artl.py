# =============================================================================
# Author: Shuo Zhou, szhou20@sheffield.ac.uk, The University of Sheffield
# =============================================================================
import numpy as np
from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from ..utils import infer_backend, lap_norm, mmd_coef, to_backend, to_numpy
from ..utils.multiclass import score2pred
from .base import BaseFramework

# =============================================================================
# Adaptation Regularisation Transfer Learning: ARTL
# Ref: Long, M., Wang, J., Ding, G., Pan, S.J. and Philip, S.Y., 2013.
# Adaptation regularization: A general framework for transfer learning.
# IEEE Transactions on Knowledge and Data Engineering, 26(5), pp.1076-1089.
# =============================================================================


def _init_artl(Xs, ys, Xt=None, yt=None, **kwargs):
    """[summary]

    Parameters
    ----------
    Xs : array-like
        Source data, shape (ns_samples, n_features)
    ys : array-like
        Source labels, shape (ns_samples,)
    Xt : array-like
        Target data, shape (nt_samples, n_features), the first ntl
        samples are labelled if yt is not None
    yt : array-like, optional
        Target label, shape (ntl_samples, ), by default None

    Returns
    -------
    X : array-like
        [description]
    y : array-like

    ker_x : array-like

    M : array-like

    unit_mat : array-like

    """

    Xs = to_numpy(Xs)
    ys = to_numpy(ys)
    Xt = to_numpy(Xt)
    yt = to_numpy(yt)
    if type(Xt) is np.ndarray:
        X = np.concatenate([Xs, Xt], axis=0)
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        M = mmd_coef(ns, nt, ys, yt, kind="joint")
    else:
        X = Xs.copy()
        M = np.zeros((X.shape[0], X.shape[0]))

    if yt is not None:
        y = np.concatenate([ys, yt])
    else:
        y = ys.copy()
    n = X.shape[0]
    ker_x = pairwise_kernels(X, **kwargs)
    ker_x[np.isnan(ker_x)] = 0
    unit_mat = np.eye(n)

    return X, y, ker_x, M, unit_mat


class ARSVM(BaseFramework):
    def __init__(
        self,
        C=1.0,
        kernel="linear",
        lambda_=1.0,
        gamma_=0.0,
        k_neighbour=5,
        solver="osqp",
        manifold_metric="cosine",
        knn_mode="distance",
        **kwargs,
    ):
        """Adaptation Regularised Support Vector Machine

        Parameters
        ----------
        C : float, optional
            param for importance of slack variable, by default 1.0
        kernel : str, optional
            'rbf' | 'linear' | 'poly' , by default 'linear'
        lambda_ : float, optional
            MMD regulisation param, by default 1.0
        gamma_ : float, optional
            manifold regulisation param, by default 0.0
        k_neighbour : int, optional
            number of nearest numbers for each sample in manifold regularisation,
            by default 5
        solver : str, optional
            solver to solve quadprog, osqp or cvxopt, by default 'osqp'
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each
            sample point. The DistanceMetric class gives a list of available
            metrics. By default 'cosine'.
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of
            returned matrix: ‘connectivity’ will return the connectivity
            matrix with ones and zeros, and ‘distance’ will return the
            distances between neighbors according to the given metric.
        kwargs :
            kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.C = C
        self.gamma_ = gamma_
        self.solver = solver
        self.k_neighbour = k_neighbour
        # self.alpha = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)
        # self.scaler = StandardScaler()

    def fit(self, Xs, ys, Xt=None, yt=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_features)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_features), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        self.backend_ = infer_backend(Xs, ys, Xt, yt)
        Xs = to_numpy(Xs)
        ys = to_numpy(ys)
        Xt = to_numpy(Xt)
        yt = to_numpy(yt)
        X, y, ker_x, M, unit_mat = _init_artl(Xs, ys, Xt, yt, metric=self.kernel, filter_params=True, **self.kwargs)

        y_ = self._lb.fit_transform(y)

        if self.gamma_ != 0:
            lap_mat = lap_norm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode)
            Q_ = unit_mat + multi_dot([(self.lambda_ * M + self.gamma_ * lap_mat), ker_x])
        else:
            Q_ = unit_mat + multi_dot([(self.lambda_ * M), ker_x])

        self.coef_, self.support_ = self._solve_semi_dual(ker_x, y_, Q_, self.C, self.solver)
        # if self._lb.y_type_ == 'binary':
        #     self.support_vectors_ = X[:nl, :][self.support_]
        #     self.n_support_ = self.support_vectors_.shape[0]
        # else:
        #     self.support_vectors_ = []
        #     self.n_support_ = []
        #     for i in range(y_.shape[1]):
        #         self.support_vectors_.append(X[:nl, :][self.support_[i]][-1])
        #         self.n_support_.append(self.support_vectors_[-1].shape[0])

        self.X = X
        self.y = y

        return self

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like
            shape (n_samples, n_features)

        Returns
        -------
        array-like
            decision scores, , shape (n_samples,) for binary classification,
            (n_samples, n_class) for multi-class cases
        """
        check_is_fitted(self, "X")
        check_is_fitted(self, "y")
        # X_fit = self.X
        backend = infer_backend(X)
        X_np = to_numpy(X)
        ker_x = pairwise_kernels(X_np, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        scores = np.dot(ker_x, self.coef_)
        return to_backend(scores, backend, reference=X)  # +self.intercept_

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like
            shape (n_samples, n_features)

        Returns
        -------
        array-like
            predicted labels, , shape (n_samples, )
        """
        backend = infer_backend(X)
        dec = to_numpy(self.decision_function(X))
        if self._lb.y_type_ == "binary":
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        y_pred = self._lb.inverse_transform(to_numpy(y_pred_))
        return to_backend(y_pred, backend, reference=X)

    def fit_predict(self, Xs, ys, Xt=None, yt=None):
        """Fit the model according to the given training data and then perform
            classification on samples in Xt.

        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_features)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_features), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        backend = infer_backend(Xs, ys, Xt, yt)
        self.fit(Xs, ys, Xt, yt)
        y_pred = self.predict(self.X)
        return to_backend(to_numpy(y_pred), backend, reference=Xs if backend == "torch" else None)


class ARRLS(BaseFramework):
    def __init__(
        self,
        kernel="linear",
        lambda_=1.0,
        gamma_=0.0,
        sigma_=1.0,
        k_neighbour=5,
        manifold_metric="cosine",
        knn_mode="distance",
        **kwargs,
    ):
        """Adaptation Regularised Least Square

        Parameters
        ----------
        kernel : str, optional
            'rbf' | 'linear' | 'poly', by default 'linear'
        lambda_ : float, optional
            MMD regularisation param, by default 1.0
        gamma_ : float, optional
            manifold regularisation param, by default 0.0
        sigma_ : float, optional
            l2 regularisation param, by default 1.0
        k_neighbour : int, optional
            number of nearest numbers for each sample in manifold regularisation,
            by default 5
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each
            sample point. The DistanceMetric class gives a list of available
            metrics. By default 'cosine'.
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of
            returned matrix: ‘connectivity’ will return the connectivity
            matrix with ones and zeros, and ‘distance’ will return the
            distances between neighbors according to the given metric.
        kwargs:
            kernel param
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.gamma_ = gamma_
        self.sigma_ = sigma_
        self.k_neighbour = k_neighbour
        # self.coef_ = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, Xs, ys, Xt=None, yt=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_features)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_features), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        self.backend_ = infer_backend(Xs, ys, Xt, yt)
        Xs = to_numpy(Xs)
        ys = to_numpy(ys)
        Xt = to_numpy(Xt)
        yt = to_numpy(yt)
        X, y, ker_x, M, unit_mat = _init_artl(Xs, ys, Xt, yt, metric=self.kernel, filter_params=True, **self.kwargs)
        n = ker_x.shape[0]
        nl = y.shape[0]
        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.gamma_ != 0:
            lap_mat = lap_norm(X, n_neighbour=self.k_neighbour, metric=self.manifold_metric, mode=self.knn_mode)
            Q_ = np.dot((J + self.lambda_ * M + self.gamma_ * lap_mat), ker_x) + self.sigma_ * unit_mat
        else:
            Q_ = np.dot((J + self.lambda_ * M), ker_x) + self.sigma_ * unit_mat

        y_ = self._lb.fit_transform(y)
        self.coef_ = self._solve_semi_ls(Q_, y_)

        self.X = X
        self.y = y

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters:
        ----------
            X : array-like,
                shape (n_samples, n_features)
        Returns
        -------
        array-like
            predicted labels, shape (n_samples)
        """
        backend = infer_backend(X)
        dec = to_numpy(self.decision_function(X))
        if self._lb.y_type_ == "binary":
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        y_pred = self._lb.inverse_transform(to_numpy(y_pred_))
        return to_backend(y_pred, backend, reference=X)

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
            X : array-like,
                shape (n_samples, n_features)
        Returns
        -------
        array-like
            prediction scores, shape (n_samples)
        """
        backend = infer_backend(X)
        X_np = to_numpy(X)
        ker_x = pairwise_kernels(X_np, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        scores = np.dot(ker_x, self.coef_)
        return to_backend(scores, backend, reference=X)

    def fit_predict(self, Xs, ys, Xt=None, yt=None):
        """Fit the model according to the given training data and then perform
            classification on samples in Xt.

        Parameters
        ----------
        Xs : array-like
            Source data, shape (ns_samples, n_features)
        ys : array-like
            Source labels, shape (ns_samples,)
        Xt : array-like
            Target data, shape (nt_samples, n_features), the first ntl
            samples are labelled if yt is not None
        yt : array-like, optional
            Target label, shape (ntl_samples, ), by default None
        """
        backend = infer_backend(Xs, ys, Xt, yt)
        self.fit(Xs, ys, Xt, yt)
        y_pred = self.predict(Xt)
        return to_backend(to_numpy(y_pred), backend, reference=Xt if Xt is not None else Xs)
