# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk, The University of Sheffield
# =============================================================================
import numpy as np
from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted

from kalelinear.estimator.base import BaseDomainAdaptationEstimator
from kalelinear.utils import lap_norm, mmd_coef, to_numpy

# =============================================================================
# Adaptation Regularisation Transfer Learning: ARTL
# Ref: Long, M., Wang, J., Ding, G., Pan, S.J. and Philip, S.Y., 2013.
# Adaptation regularization: A general framework for transfer learning.
# IEEE Transactions on Knowledge and Data Engineering, 26(5), pp.1076-1089.
# =============================================================================


def _init_artl(estimator, Xs, ys, Xt=None, yt=None, **kwargs):
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

    x_kernel_matrix : array-like

    M : array-like

    unit_matrix : array-like

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
    X, y, _, x_kernel_matrix, unit_matrix, _, _ = estimator._prepare_kernel_fit_data(X, y, **kwargs)

    return X, y, x_kernel_matrix, M, unit_matrix


def _uses_legacy_artl_inputs(X, covariates=None, Xt=None):
    if Xt is not None or covariates is None:
        return True

    x = to_numpy(X)
    covariates = np.asarray(to_numpy(covariates))
    return covariates.ndim == 0 or covariates.shape[0] != x.shape[0]


def _prepare_artl_fit_data(
    estimator,
    X,
    y,
    covariates=None,
    target_covariate=None,
    unlabeled_value=None,
    Xt=None,
    yt=None,
    **kwargs,
):
    if _uses_legacy_artl_inputs(X, covariates=covariates, Xt=Xt):
        if Xt is None and covariates is not None:
            Xt = covariates
        estimator.source_idx_ = None
        estimator.target_idx_ = None
        estimator.target_fit_idx_ = None
        estimator.target_covariate_ = target_covariate
        return _init_artl(estimator, X, y, Xt, yt, **kwargs)

    split = estimator._split_source_target_by_covariate(
        X,
        y,
        covariates,
        target_covariate=target_covariate,
        unlabeled_value=unlabeled_value,
    )
    estimator.source_idx_ = split["source_idx"]
    estimator.target_idx_ = split["target_idx"]
    estimator.target_fit_idx_ = split["target_fit_idx"]
    estimator.target_covariate_ = split["target_covariate"]
    return _init_artl(estimator, split["Xs"], split["ys"], split["Xt"], split["yt"], **kwargs)


class ARSVM(BaseDomainAdaptationEstimator):
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

    def fit(self, X, y, covariates=None, target_covariate=None, unlabeled_value=None, Xt=None, yt=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Source and target data, shape (n_samples, n_features). For
            backwards compatibility, this may also be source data when ``Xt``
            is provided.
        y : array-like
            Source labels only, or one label per row in ``X``. If full-length
            labels include unlabeled target rows, mark them with
            ``unlabeled_value``.
        covariates : array-like, optional
            Binary domain labels aligned with ``X``. The value specified by
            ``target_covariate`` identifies target rows; all other rows are
            treated as source rows.
        target_covariate : scalar, optional
            Domain value identifying target samples. Defaults to the last
            sorted unique covariate value.
        unlabeled_value : scalar, optional
            Sentinel used for unlabeled target rows when ``y`` is full length.
        Xt : array-like, optional
            Legacy target data.
        yt : array-like, optional
            Legacy target labels.
        """
        X, y, x_kernel_matrix, M, unit_matrix = _prepare_artl_fit_data(
            self,
            X,
            y,
            covariates=covariates,
            target_covariate=target_covariate,
            unlabeled_value=unlabeled_value,
            Xt=Xt,
            yt=yt,
            metric=self.kernel,
            filter_params=True,
            **self.kwargs,
        )

        y_ = self._lb.fit_transform(y)

        if self.gamma_ != 0:
            lap_mat = lap_norm(X, n_neighbour=self.k_neighbour, mode=self.knn_mode)
            Q_ = unit_matrix + multi_dot([(self.lambda_ * M + self.gamma_ * lap_mat), x_kernel_matrix])
        else:
            Q_ = unit_matrix + multi_dot([(self.lambda_ * M), x_kernel_matrix])

        self.coef_, self.support_ = self._solve_semi_dual(x_kernel_matrix, y_, Q_, self.C, self.solver)
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
            (n_samples, n_classes) for multi-class cases
        """
        check_is_fitted(self, "X")
        check_is_fitted(self, "y")
        # x_fit = self.X
        x_np = to_numpy(X)
        x_kernel_matrix = pairwise_kernels(x_np, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        scores = np.dot(x_kernel_matrix, self.coef_)
        return scores  # +self.intercept_

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
        dec = to_numpy(self.decision_function(X))
        return self._lb.inverse_transform(dec, threshold=0)

    def fit_predict(self, X, y, covariates=None, target_covariate=None, unlabeled_value=None, Xt=None, yt=None):
        """Fit the model according to the given training data and then perform
            classification on target samples.

        Parameters
        ----------
        X : array-like
            Combined source and target data, or legacy source data when
            ``Xt`` is provided.
        y : array-like
            Source labels or full-length labels.
        covariates : array-like, optional
            Binary domain labels aligned with ``X``.
        """
        legacy_inputs = _uses_legacy_artl_inputs(X, covariates=covariates, Xt=Xt)
        legacy_target = Xt if Xt is not None else covariates

        self.fit(
            X,
            y,
            covariates=covariates,
            target_covariate=target_covariate,
            unlabeled_value=unlabeled_value,
            Xt=Xt,
            yt=yt,
        )

        if legacy_inputs:
            predict_X = legacy_target if legacy_target is not None else self.X
        else:
            predict_X = to_numpy(X)[self.target_idx_]

        return self.predict(predict_X)


class ARRLS(BaseDomainAdaptationEstimator):
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

    def fit(self, X, y, covariates=None, target_covariate=None, unlabeled_value=None, Xt=None, yt=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like
            Source and target data, shape (n_samples, n_features). For
            backwards compatibility, this may also be source data when ``Xt``
            is provided.
        y : array-like
            Source labels only, or one label per row in ``X``. If full-length
            labels include unlabeled target rows, mark them with
            ``unlabeled_value``.
        covariates : array-like, optional
            Binary domain labels aligned with ``X``.
        target_covariate : scalar, optional
            Domain value identifying target samples. Defaults to the first
            sorted covariate value.
        unlabeled_value : scalar, optional
            Sentinel used for unlabeled target rows when ``y`` is full length.
        Xt : array-like, optional
            Legacy target data.
        yt : array-like, optional
            Legacy target labels.
        """
        X, y, x_kernel_matrix, M, unit_matrix = _prepare_artl_fit_data(
            self,
            X,
            y,
            covariates=covariates,
            target_covariate=target_covariate,
            unlabeled_value=unlabeled_value,
            Xt=Xt,
            yt=yt,
            metric=self.kernel,
            filter_params=True,
            **self.kwargs,
        )
        n = x_kernel_matrix.shape[0]
        nl = y.shape[0]
        J = np.zeros((n, n))
        J[:nl, :nl] = np.eye(nl)

        if self.gamma_ != 0:
            lap_mat = lap_norm(X, n_neighbour=self.k_neighbour, metric=self.manifold_metric, mode=self.knn_mode)
            Q_ = np.dot((J + self.lambda_ * M + self.gamma_ * lap_mat), x_kernel_matrix) + self.sigma_ * unit_matrix
        else:
            Q_ = np.dot((J + self.lambda_ * M), x_kernel_matrix) + self.sigma_ * unit_matrix

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
        dec = to_numpy(self.decision_function(X))
        return self._lb.inverse_transform(dec, threshold=0)

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
        x_np = to_numpy(X)
        x_kernel_matrix = pairwise_kernels(x_np, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        scores = np.dot(x_kernel_matrix, self.coef_)
        return scores

    def fit_predict(self, X, y, covariates=None, target_covariate=None, unlabeled_value=None, Xt=None, yt=None):
        """Fit the model according to the given training data and then perform
            classification on target samples.

        Parameters
        ----------
        X : array-like
            Combined source and target data, or legacy source data when
            ``Xt`` is provided.
        y : array-like
            Source labels or full-length labels.
        covariates : array-like, optional
            Binary domain labels aligned with ``X``.
        """
        legacy_inputs = _uses_legacy_artl_inputs(X, covariates=covariates, Xt=Xt)
        legacy_target = Xt if Xt is not None else covariates

        self.fit(
            X,
            y,
            covariates=covariates,
            target_covariate=target_covariate,
            unlabeled_value=unlabeled_value,
            Xt=Xt,
            yt=yt,
        )

        if legacy_inputs:
            predict_X = legacy_target if legacy_target is not None else self.X
        else:
            predict_X = to_numpy(X)[self.target_idx_]

        return self.predict(predict_X)
