"""
@author: Shuo Zhou, The University of Sheffield, shuo.zhou@sheffield.ac.uk

References
----------
Zhou, S., Li, W., Cox, C.R. and Lu, H., 2020. Side Information Dependence as a
Regulariser for Analyzing Human Brain Conditions across Cognitive Experiments.
In Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI 2020).
"""

import numpy as np
from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer

from kalelinear._covariates import check_numeric_covariates, fit_covariate_encoder

from ..utils import infer_backend, lap_norm, to_backend, to_numpy

# import cvxpy as cvx
# from cvxpy.error import SolverError
from ..utils.multiclass import score2pred
from .base import BaseDomainAdaptationEstimator


def _fit_model_covariates(estimator, covariates, n_samples):
    covariates, encoder = fit_covariate_encoder(
        covariates,
        estimator.covariate_encoder,
        n_samples,
        error_prefix="Covariates",
    )
    estimator.covariate_encoder_ = encoder
    return check_numeric_covariates(
        covariates,
        n_samples,
        error_prefix="Covariates",
        numeric_error_message=(
            "Covariates must be numeric when `covariate_encoder` is None. "
            "Provide numeric covariates or set `covariate_encoder`."
        ),
    )


class CoIRSVM(BaseDomainAdaptationEstimator):
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
        covariate_encoder=None,
        **kwargs,
    ):
        """Covariate Independence Regularised Support Vector Machine

        Parameters
        ----------
        C : float, optional
            param for importance of slack variable, by default 1
        kernel : str, optional
            'rbf' | 'linear' | 'poly', by default 'linear'
        lambda_ : float, optional
            param for covariate (side information) independence regularisation, by default 1
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
            {'connectivity', 'distance'}, by default 'distance'. Type of
            returned matrix: 'connectivity' will return the connectivity
            matrix with ones and zeros, and 'distance' will return the
            distances between neighbors according to the given metric.
        solver : str, optional
            quadratic programming solver, [cvxopt, osqp], by default 'osqp'
        """
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.C = C
        self.solver = solver
        # self.scaler = StandardScaler()
        # self.coef_ = None
        # self.X = None
        # self.y = None
        # self.support_ = None
        # self.support_vectors_ = None
        # self.n_support_ = None
        self.manifold_metric = manifold_metric
        self.k_neighbour = k_neighbour
        self.knn_mode = knn_mode
        self.covariate_encoder = covariate_encoder
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)

    def fit(self, X, y, covariates=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label, shape (n_labeled_samples, ) where n_labeled_samples <= n_samples
        covariates : array-like, optional
            Domain covariate matrix for input data, shape (n_samples, n_covariates).
            If ``None``, the covariate-independence term is disabled, so with
            ``mu=0`` the objective degenerates to the standard kernel SVM form.

        Returns
        -------
        self
            [description]
        """
        self.backend_ = infer_backend(X, y, covariates)
        X, y, covariates, x_kernel_matrix, unit_matrix, centering_matrix, n = self._prepare_kernel_fit_data(
            X, y, covariates, kernel=self.kernel, **self.kwargs
        )
        covariates = _fit_model_covariates(self, covariates, n)
        if isinstance(covariates, np.ndarray):
            c_kernel_matrix = np.dot(covariates, covariates.T)
        else:
            c_kernel_matrix = np.zeros((n, n))
        y_ = self._lb.fit_transform(y)

        Q_ = unit_matrix.copy()
        if self.mu != 0:
            lap_mat = lap_norm(X, n_neighbour=self.k_neighbour, metric=self.manifold_metric, mode=self.knn_mode)
            Q_ += np.dot(
                self.lambda_ / np.square(n - 1) * multi_dot([centering_matrix, c_kernel_matrix, centering_matrix])
                + self.mu / np.square(n) * lap_mat,
                x_kernel_matrix,
            )
        else:
            Q_ += (
                self.lambda_
                * multi_dot([centering_matrix, c_kernel_matrix, centering_matrix, x_kernel_matrix])
                / np.square(n - 1)
            )

        self.coef_, self.support_ = self._solve_semi_dual(x_kernel_matrix, y_, Q_, self.C, self.solver)

        # if self._lb.y_type_ == 'binary':
        #     self.coef_, self.support_ = self._semi_binary_dual(K, y_, Q_,
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
        #     for i in range(y_.shape[1]):
        #         coef_, support_ = self._semi_binary_dual(K, y_[:, i], Q_,
        #                                                  self.C,
        #                                                  self.solver)
        #         coef_list.append(coef_.reshape(-1, 1))
        #         self.support_.append(support_)
        #         self.support_vectors_.append(X[:nl, :][support_][-1])
        #         self.n_support_.append(self.support_vectors_[-1].shape[0])
        #     self.coef_ = np.concatenate(coef_list, axis=1)

        self.X = X
        self.y = y

        return self

    def decision_function(self, X):
        """Decision function for the samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)

        Returns
        -------
        array-like
            decision scores, shape (n_samples,) for binary classification,
            (n_samples, n_classes) for multi-class cases
        """
        backend = infer_backend(X)
        x_np = to_numpy(X)
        x_kernel_matrix = pairwise_kernels(x_np, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        scores = np.dot(x_kernel_matrix, self.coef_)
        return to_backend(scores, backend, reference=X)  # +self.intercept_

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
        backend = infer_backend(X)
        dec = to_numpy(self.decision_function(X))
        if self._lb.y_type_ == "binary":
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        y_pred = self._lb.inverse_transform(to_numpy(y_pred_))
        return to_backend(y_pred, backend, reference=X)

    def fit_predict(self, X, y, covariates):
        """Fit the model according to the given training data and then perform classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label, shape (n_labeled_samples, ) where n_labeled_samples <= n_samples
        covariates : array-like, optional
            Domain covariate matrix for input data, shape (n_samples, n_covariates).
            If ``None``, the covariate-independence term is disabled, so with
            ``mu=0`` the objective degenerates to the standard kernel least-squares form.

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        self.fit(X, y, covariates)
        return self.predict(X)


class CoIRLS(BaseDomainAdaptationEstimator):
    def __init__(
        self,
        sigma_=1.0,
        lambda_=1.0,
        mu=0.0,
        kernel="linear",
        k=3,
        knn_mode="distance",
        manifold_metric="cosine",
        class_weight=None,
        covariate_encoder=None,
        **kwargs,
    ):
        """Covariate Independence Regularised Least Square

        Parameters
        ----------
        sigma_ : float, optional
            param for model complexity (l2 norm), by default 1.0
        lambda_ : float, optional
            param for covariate (side information) independence regularisation, by default 1.0
        mu : float, optional
            param for manifold regularisation, by default 0.0
        kernel : str, optional
            [description], by default 'linear'
        k : int, optional
            number of nearest numbers for each sample in manifold regularisation,
            by default 3
        knn_mode : str, optional
            {'connectivity', 'distance'}, by default 'distance'. Type of
            returned matrix: 'connectivity' will return the connectivity
            matrix with ones and zeros, and 'distance' will return the
            distances between neighbors according to the given metric.
        manifold_metric : str, optional
            The distance metric used to calculate the k-Neighbors for each
            sample point. The DistanceMetric class gives a list of available
            metrics. By default 'cosine'.
        class_weight : [type], optional
            [description], by default None
        **kwargs:
            kernel param
        """
        self.kernel = kernel
        self.sigma_ = sigma_
        self.lambda_ = lambda_
        self.mu = mu
        # self.classes = None
        # self.coef_ = None
        # self.X = None
        # self.y = None
        self.manifold_metric = manifold_metric
        self.k = k
        self.knn_mode = knn_mode
        self.class_weight = class_weight
        self.covariate_encoder = covariate_encoder
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)
        self.kwargs = kwargs

    def fit(self, X, y, covariates=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label, shape (n_labeled_samples, ) where n_labeled_samples <= n_samples
        covariates : array-like,
            Domain covariate matrix for input data, shape (n_samples, n_covariates)

        Returns
        -------
        self
            [description]
        """
        self.backend_ = infer_backend(X, y, covariates)
        X, y, covariates, x_kernel_matrix, unit_matrix, centering_matrix, n = self._prepare_kernel_fit_data(
            X, y, covariates, kernel=self.kernel, **self.kwargs
        )
        covariates = _fit_model_covariates(self, covariates, n)
        # X, D = cat_data(Xl, Dl, Xu, Du)
        nl = y.shape[0]
        if isinstance(covariates, np.ndarray):
            c_kernel_matrix = np.dot(covariates, covariates.T)
        else:
            c_kernel_matrix = np.zeros((n, n))

        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.mu != 0:
            lap_mat = lap_norm(X, n_neighbour=self.k, mode=self.knn_mode, metric=self.manifold_metric)
            Q_ = self.sigma_ * unit_matrix + np.dot(
                J
                + self.lambda_ / np.square(n - 1) * multi_dot([centering_matrix, c_kernel_matrix, centering_matrix])
                + self.mu / np.square(n) * lap_mat,
                x_kernel_matrix,
            )
        else:
            Q_ = self.sigma_ * unit_matrix + np.dot(
                J + self.lambda_ / np.square(n - 1) * multi_dot([centering_matrix, c_kernel_matrix, centering_matrix]),
                x_kernel_matrix,
            )

        y_ = self._lb.fit_transform(y)
        self.coef_ = self._solve_semi_ls(Q_, y_)

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
            (n_samples, n_classes) for multi-class cases
        """
        backend = infer_backend(X)
        x_np = to_numpy(X)
        x_kernel_matrix = pairwise_kernels(x_np, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        scores = np.dot(x_kernel_matrix, self.coef_)
        return to_backend(scores, backend, reference=X)  # +self.intercept_

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
        backend = infer_backend(X)
        dec = to_numpy(self.decision_function(X))
        if self._lb.y_type_ == "binary":
            y_pred_ = np.sign(dec).reshape(-1, 1)
        else:
            y_pred_ = score2pred(dec)

        y_pred = self._lb.inverse_transform(to_numpy(y_pred_))
        return to_backend(y_pred, backend, reference=X)

    def fit_predict(self, X, y, covariates=None):
        """Fit the model according to the given training data and then perform
            classification on samples in X.

        Parameters
        ----------
        X : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Label, shape (n_labeled_samples, ) where n_labeled_samples <= n_samples
        covariates : array-like,
            Domain covariate matrix for input data, shape (n_samples, n_covariates)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        self.fit(X, y, covariates)
        return self.predict(X)
