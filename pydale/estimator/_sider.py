"""
@author: Shuo Zhou, The University of Sheffield, szhou@sheffield.ac.uk

References
----------
Zhou, S., Li, W., Cox, C.R. and Lu, H., 2020. Side Information Dependence as a
Regulariser for Analyzing Human Brain Conditions across Cognitive Experiments.
In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020).
"""

import numpy as np
from numpy.linalg import multi_dot

# import cvxpy as cvx
# from cvxpy.error import SolverError
from ..utils import base_init, lap_norm
from .base import BaseFramework


class SIDeRSVM(BaseFramework):
    def __init__(
        self,
        C=1.0,
        kernel="linear",
        lambda_=1.0,
        mu=0.0,
        k_neighbour=3,
        manifold_metric="cosine",
        knn_mode="distance",
        solver="osqp",
        **kwargs,
    ):
        """Side Information Dependence Regularised Support Vector Machine

        Parameters
        ----------
        C : float, optional
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be
            strictly positive. The penalty is a squared l2 penalty, by default 1
        kernel : str, optional
            'rbf' | 'linear' | 'poly', by default 'linear'
        lambda_ : float, optional
            param for side information dependence regularisation, by default 1
        mu : float, optional
            param for manifold regularisation, by default 0
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
        solver : str, optional
            quadratic programming solver, [cvxopt, osqp], by default 'osqp'
        """
        super().__init__(kernel, k_neighbour, manifold_metric, knn_mode, **kwargs)
        self.lambda_ = lambda_
        self.mu = mu
        self.C = C
        self.solver = solver
        # self.support_vectors_ = None
        self.support_ = None

    def fit(self, x, y, covariates=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label,, shape (n_labeled_samples, ) where n_labeled_samples <= n_samples
        covariates : array-like,
            Domain co-variate matrix for input data, shape (n_samples, n_co-variates)

        Returns
        -------
        self
            [description]
        """
        krnl_x, unit_mat, ctr_mat, n = base_init(x, kernel=self.kernel, **self.kwargs)
        krnl_c = np.dot(covariates, covariates.T)
        y_transformed = self._lb.fit_transform(y)

        obj_core = unit_mat.copy()  # core objective
        if self.mu != 0:
            lap_mat = lap_norm(x, n_neighbour=self.k_neighbour, metric=self.manifold_metric, mode=self.knn_mode,)
            obj_core += np.dot(
                self.lambda_ * multi_dot([ctr_mat, krnl_c, ctr_mat]) / np.square(n - 1)
                + self.mu * lap_mat / np.square(n),
                krnl_x,
            )
        else:
            obj_core += self.lambda_ * multi_dot([ctr_mat, krnl_c, ctr_mat, krnl_x]) / np.square(n - 1)

        self.coef_, self.support_ = self._solve_semi_dual(krnl_x, y_transformed, obj_core, self.C, self.solver)

        # if self._lb.y_type_ == 'binary':
        #     self.coef_, self.support_ = self._semi_binary_dual(K, y_transformed, obj_core,
        #                                                        self.C,
        #                                                        self.solver)
        #     self.support_vectors_ = X[:nl, :][self.support_]
        #     self.n_support_ = self.support_vectors_.shape[0]
        #
        # else:
        #     coef_list = []
        #     self.support_ = []
        #     self.support_vectors_ = []
        #     self.n_support_ = []
        #     for i in range(y_transformed.shape[1]):
        #         coef_, support_ = self._semi_binary_dual(K, y_transformed[:, i], obj_core,
        #                                                  self.C,
        #                                                  self.solver)
        #         coef_list.append(coef_.reshape(-1, 1))
        #         self.support_.append(support_)
        #         self.support_vectors_.append(X[:nl, :][support_][-1])
        #         self.n_support_.append(self.support_vectors_[-1].shape[0])
        #     self.coef_ = np.concatenate(coef_list, axis=1)

        self.x = x

        return self

    def fit_predict(self, x, y, covariates):
        """[summary]

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label, shape (n_labeled_samples, ) where n_labeled_samples <= n_samples
        covariates : array-like,
            Domain co-variate matrix for input data, shape (n_samples, n_co-variates)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        self.fit(x, y, covariates)
        return self.predict(x)


class SIDeRLS(BaseFramework):
    def __init__(
        self,
        sigma_=1.0,
        lambda_=1.0,
        mu=0.0,
        kernel="linear",
        k_neighbour=3,
        knn_mode="distance",
        manifold_metric="cosine",
        class_weight=None,
        **kwargs,
    ):
        """Side Information Dependence Regularised Least Square

        Parameters
        ----------
        sigma_ : float, optional
            param for model complexity (l2 norm), by default 1.0
        lambda_ : float, optional
            param for side information dependence regularisation, by default 1.0
        mu : float, optional
            param for manifold regularisation, by default 0.0
        kernel : str, optional
            [description], by default 'linear'
        k_neighbour : int, optional
            number of nearest numbers for each sample in manifold regularisation,
            by default 3
        knn_mode : str, optional
            {‘connectivity’, ‘distance’}, by default 'distance'. Type of returned matrix: ‘connectivity’ will return
            the connectivity matrix with ones and zeros, and ‘distance’ will return the distances between neighbors
            according to the given metric.
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each sample point. The DistanceMetric class
            gives a list of available metrics. By default 'cosine'.
        class_weight : [type], optional
            [description], by default None
        **kwargs:
            kernel param
        """
        super().__init__(kernel, k_neighbour, manifold_metric, knn_mode, **kwargs)
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self.mu = mu
        self.class_weight = class_weight

    def fit(self, x, y, covariates=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples
        covariates : array-like,
            Domain co-variate matrix for input data, shape (n_samples, n_co-variates)

        Returns
        -------
        self
            [description]
        """
        n_labeled = y.shape[0]
        krnl_x, unit_mat, ctr_mat, n = base_init(x, kernel=self.kernel, **self.kwargs)
        if type(covariates) == np.ndarray:
            krnl_c = np.dot(covariates, covariates.T)
        else:
            krnl_c = np.zeros((n, n))

        J = np.zeros((n, n))
        J[:n_labeled, :n_labeled] = np.eye(n_labeled)

        Q = self.sigma_ * unit_mat
        if self.mu != 0:
            lap_mat = lap_norm(x, n_neighbour=self.k, metric=self.manifold_metric, mode=self.knn_mode)
            Q += np.dot(
                J
                + self.lambda_ * multi_dot([ctr_mat, krnl_c, ctr_mat]) / np.square(n - 1)
                + self.mu * lap_mat / np.square(n),
                krnl_x,
            )
        else:
            Q += np.dot(J + self.lambda_ * multi_dot([ctr_mat, krnl_c, ctr_mat]) / np.square(n - 1), krnl_x,)

        y_transformed = self._lb.fit_transform(y)
        self.coef_ = self._solve_semi_ls(Q, y_transformed)

        self.x = x

        return self

    def fit_predict(self, x, y, covariates=None):
        """Fit the model according to the given training data and then perform classification on samples in X.

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label,, shape (nl_samples, ) where nl_samples <= n_samples
        covariates : array-like,
            Domain co-variate matrix for input data, shape (n_samples, n_co-variates)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        self.fit(x, y, covariates)
        return self.predict(x)
