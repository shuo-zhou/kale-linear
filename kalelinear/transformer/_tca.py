# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer

from kalelinear.utils import base_init, infer_backend, lap_norm, mmd_coef, to_backend, to_numpy

# from sklearn.utils.multiclass import type_of_target
# from sklearn.utils.validation import check_is_fitted


class TCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel="linear", lambda_=1.0, mu=1.0, gamma_=0.5, k=3, **kwargs):
        """Transfer Component Analysis: TCA

        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= (N, d))
        kernel: str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        lambda_ : float
            domain divergence regularisation param
        mu : float
            KNN graph param
        k : int
            number of nearest neighbour for KNN graph
        gamma : float
            label dependence param

        References
        ----------
        S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via
        Transfer Component Analysis," in IEEE Transactions on Neural Networks,
        vol. 22, no. 2, pp. 199-210, Feb. 2011.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu
        self.gamma_ = gamma_
        self.k = k
        self._lb = LabelBinarizer(pos_label=1, neg_label=0)

    def fit(self, X, y=None, covariates=None, target_covariate=None):
        """[summary]
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.

        Parameters
        ----------

        """

        n_samples, n_features = X.shape
        target_idx = None
        source_idx = None
        # degenerate to normal kernel PCA if covariates are not given
        if covariates is None:
            L = np.zeros((n_samples, n_samples))
            ns = n_samples
        else:
            # check if covariates is an array with binary values
            if not np.array_equal(covariates, covariates.astype(bool)):
                raise ValueError("Covariates for TCA should be binary values.")
            if target_covariate is None:
                target_covariate = covariates[0]
            target_idx = np.where(covariates == target_covariate)[0]
            source_idx = np.where(covariates != target_covariate)[0]
            Xt = X[target_idx, :]
            Xs = X[source_idx, :]
            ns = Xs.shape[0]
            nt = Xt.shape[0]
            L = mmd_coef(ns, nt, kind="marginal", mu=0)
            L[np.isnan(L)] = 0
            X = np.vstack((Xs, Xt))

        x_kernel_matrix, unit_matrix, centering_matrix, n = base_init(X, kernel=self.kernel, **self.kwargs)

        obj = self.lambda_ * unit_matrix
        st = multi_dot([x_kernel_matrix, centering_matrix, x_kernel_matrix.T])

        # check if y is an numpy array, if not, degenerate to unsupervised TCA

        if isinstance(y, np.ndarray):
            n_labeled_samples = y.shape[0]
            # check if target_idx and source_idx are defined
            if not (source_idx is not None or target_idx is not None):
                raise ValueError(
                    "Source and target indices are not defined. Please provide covariates and target_covariate."
                )
            if n_labeled_samples == ns:
                ys = y
                yt = None
            elif n_samples >= n_labeled_samples > ns:
                ys = y[source_idx]
                # yt not source idx
                yt = y[~np.isin(np.arange(n_labeled_samples), source_idx)]
            else:
                raise ValueError("Number of labeled samples does not meet the required conditions.")

            # semisupervised TCA (SSTCA)
            ys_transformed = self._lb.fit_transform(ys)
            n_classes = ys_transformed.shape[1]
            y_transformed = np.zeros((n_samples, n_classes))
            y_transformed[: ys_transformed.shape[0], :] = ys_transformed[:]
            if yt is not None:
                yt_mat = self._lb.transform(yt)
                y_transformed[ys_transformed.shape[0] : yt_mat.shape[0], :] = yt_mat[:]
            y_kernel_matrix = self.gamma_ * np.dot(y_transformed, y_transformed.T) + (1 - self.gamma_) * unit_matrix
            lap_mat = lap_norm(X, n_neighbour=self.k, mode="connectivity")
            obj += multi_dot([x_kernel_matrix, (L + self.mu * lap_mat), x_kernel_matrix.T])
            st += multi_dot([x_kernel_matrix, centering_matrix, y_kernel_matrix, centering_matrix, x_kernel_matrix.T])

        # obj = np.trace(np.dot(x_kernel_matrix, L))
        else:
            obj += multi_dot([x_kernel_matrix, L, x_kernel_matrix.T])

        eig_values, eig_vectors = eig(obj, st)
        idx_sorted = eig_values.argsort()

        self.U = np.asarray(eig_vectors[:, idx_sorted], dtype=np.float64)
        self.X = X

        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like,
            shape (n_samples, n_features)

        Returns
        -------
        array-like
            transformed data
        """
        # check_is_fitted(self, 'Xs')
        # check_is_fitted(self, 'Xt')
        backend = infer_backend(X)
        x_np = to_numpy(X)
        if hasattr(self, "X"):
            x_fit = self.X
        else:
            raise ValueError("The fit method should be called before transform.")
        x_kernel_matrix = pairwise_kernels(x_np, x_fit, metric=self.kernel, filter_params=True, **self.kwargs)
        x_transformed = np.dot(x_kernel_matrix, self.U[:, : self.n_components])
        return to_backend(x_transformed, backend, reference=X)

    def fit_transform(self, X, y=None, covariates=None, target_covariate=None):
        """
        Parameters
        ----------


        Returns
        -------
        array-like
            transformed Xs_transformed, Xt_transformed
        """
        self.fit(X, y=y, covariates=covariates, target_covariate=target_covariate)
        return self.transform(X)
