"""
@author: Shuo Zhou, The University of Sheffield, szhou@sheffield.ac.uk

Ref: Belkin, M., Niyogi, P., & Sindhwani, V. (2006). Manifold regularization:
A geometric framework for learning from labeled and unlabeled examples.
Journal of machine learning research, 7(Nov), 2399-2434.
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

# import cvxpy as cvx
# from cvxpy.error import SolverError
from ..utils import base_init, lap_norm
from ..utils.multiclass import score2pred
from .base import BaseFramework


class LapSVM(BaseFramework):
    def __init__(
        self,
        C=1.0,
        kernel="linear",
        gamma_=1.0,
        solver="osqp",
        k_neighbour=3,
        manifold_metric="cosine",
        knn_mode="distance",
        **kwargs,
    ):
        """Laplacian Regularized Support Vector Machine

        Parameters
        ----------
        C : float, optional
            param for importance of slack variable, by default 1.0
        kernel : str, optional
            'rbf' | 'linear' | 'poly', by default 'linear'
        gamma_ : float, optional
            param for manifold regularisation, by default 1.0
        solver : str, optional
            quadratic programming solver, [cvxopt, osqp], by default 'osqp'
        k_neighbour : int, optional
            number of nearest numbers for each sample in manifold regularisation,
            by default 3
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each
            sample point. The DistanceMetric class gives a list of available
            metrics. By default 'cosine'.
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of
            returned matrix: ‘connectivity’ will return the connectivity
            matrix with ones and zeros, and ‘distance’ will return the
            distances between neighbors according to the given metric.
        **kwargs:
            kernel param
        """
        self.C = C
        self.gamma_ = gamma_
        self.kernel = kernel
        self.solver = solver
        self.kwargs = kwargs
        self.manifold_metric = manifold_metric
        self.k_neighbour = k_neighbour
        self.knn_mode = knn_mode
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples

        Returns
        -------
        self
            [description]
        """
        ker_x, unit_mat, ctr_mat, n = base_init(X, kernel=self.kernel, **self.kwargs)
        if self.gamma_ == 0:
            Q_ = ctr_mat
        else:
            lap_mat = lap_norm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode)
            Q_ = ctr_mat + self.gamma_ * np.dot(lap_mat, ker_x)

        y_ = self._lb.fit_transform(y)
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
            Input data, shape (n_samples, n_features)

        Returns
        -------
        array-like
            decision scores, shape (n_samples,) for binary classification,
            (n_samples, n_class) for multi-class cases
        """
        check_is_fitted(self, "X")
        check_is_fitted(self, "y")
        # X_fit = self.X
        ker_x = pairwise_kernels(X, self.X, metric=self.kernel, filter_params=True, **self.kwargs)

        return np.dot(ker_x, self.coef_)  # +self.intercept_

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == "binary":
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def fit_predict(self, X, y):
        """Fit the model according to the given training data and then perform
            classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        self.fit(X, y)
        return self.predict(X)


class LapRLS(BaseFramework):
    def __init__(
        self,
        kernel="linear",
        gamma_=1.0,
        sigma_=1.0,
        k_neighbour=5,
        manifold_metric="cosine",
        knn_mode="distance",
        **kwargs,
    ):
        """Laplacian Regularized Least Squares

        Parameters
        ----------
        kernel : str, optional
            'rbf' | 'linear' | 'poly', by default 'linear'
        gamma_ : float, optional
            manifold regularisation param, by default 1.0
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
            kernel params
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.gamma_ = gamma_
        self.sigma_ = sigma_
        self.k_neighbour = k_neighbour
        # self.coef_ = None
        self.knn_mode = knn_mode
        self.manifold_metric = manifold_metric
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y):
        """ "Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples

        Returns
        -------
        self
            [description]
        """
        nl = y.shape[0]
        ker_x, unit_mat, ctr_mat, n = base_init(X, kernel=self.kernel, **self.kwargs)

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.gamma_ != 0:
            lap_mat = lap_norm(X, n_neighbour=self.k_neighbour, metric=self.manifold_metric, mode=self.knn_mode)
            Q_ = np.dot((J + self.gamma_ * lap_mat), ker_x) + self.sigma_ * unit_mat
        else:
            Q_ = np.dot(J, ker_x) + self.sigma_ * ctr_mat

        y_ = self._lb.fit_transform(y)
        self.coef_ = self._solve_semi_ls(Q_, y_)

        self.X = X
        self.y = y

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        dec = self.decision_function(X)
        if self._lb.y_type_ == "binary":
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        return self._lb.inverse_transform(y_pred_)

    def decision_function(self, X):
        """Evaluates the decision function for the samples in X

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)

        Returns
        -------
        array-like
            decision scores, shape (n_samples,) for binary classification,
            (n_samples, n_class) for multi-class cases
        """
        ker_x = pairwise_kernels(X, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(ker_x, self.coef_)

    def fit_predict(self, X, y):
        """Fit the model according to the given training data and then perform
            classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        self.fit(X, y)

        return self.predict(X)
