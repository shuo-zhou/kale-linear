# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
import numpy as np
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels

from kalelinear.utils import base_init, infer_backend, mmd_coef, to_backend, to_numpy


class JDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel="linear", lambda_=1.0, mu=1.0, **kwargs):
        """Joint Distribution Adaptation.

        Parameters
        ----------
        n_components : int
            Number of projected components.
        kernel : str, default="linear"
            Kernel metric passed to :func:`sklearn.metrics.pairwise.pairwise_kernels`.
        lambda_ : float, default=1.0
            Domain divergence regularisation parameter.
        mu : float, default=1.0
            Weight of the conditional MMD term when source and target labels are available.
        **kwargs : dict
            Additional kernel parameters.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu

    def fit(self, X, y=None, covariates=None, target_covariate=None):
        """Fit the JDA transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data containing both source and target samples.
        y : array-like of shape (n_labeled_samples,), optional
            Labels for the source samples only, or for the first ``n_labeled_samples`` rows of ``X``.
        covariates : array-like of shape (n_samples,), optional
            Binary domain indicator where one value marks the target domain.
        target_covariate : scalar, optional
            Value in ``covariates`` that identifies the target domain. Defaults to ``covariates[0]``.
        """
        X = to_numpy(X)
        y = to_numpy(y) if y is not None else None
        covariates = to_numpy(covariates) if covariates is not None else None

        n_samples, _ = X.shape
        target_idx = None
        source_idx = None

        # Degenerate to kernel PCA when domain covariates are not provided.
        if covariates is None:
            L = np.zeros((n_samples, n_samples))
        else:
            covariates = np.asarray(covariates).reshape(-1)
            if covariates.shape[0] != n_samples:
                raise ValueError("Covariates and X must have the same number of samples.")
            if not np.array_equal(covariates, covariates.astype(bool)):
                raise ValueError("Covariates for JDA should be binary values.")

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

            if y is not None:
                n_labeled_samples = y.shape[0]

                if n_labeled_samples == ns:
                    ys = y
                    yt = None
                elif n_samples >= n_labeled_samples > ns:
                    ys = y[source_idx]
                    yt = y[~np.isin(np.arange(n_labeled_samples), source_idx)]
                else:
                    raise ValueError("Number of labeled samples does not meet the required conditions.")

                if yt is not None:
                    L = mmd_coef(ns, nt, ys, yt, kind="joint", mu=self.mu)
                    L[np.isnan(L)] = 0

        x_kernel_matrix, unit_matrix, centering_matrix, _ = base_init(X, kernel=self.kernel, **self.kwargs)

        obj = np.dot(np.dot(x_kernel_matrix, L), x_kernel_matrix.T) + self.lambda_ * unit_matrix
        st = np.dot(np.dot(x_kernel_matrix, centering_matrix), x_kernel_matrix.T)

        eig_values, eig_vectors = eig(obj, st)
        eig_values = np.real(eig_values)
        eig_vectors = np.real(eig_vectors)
        idx_sorted = eig_values.argsort()

        self.U = np.asarray(eig_vectors[:, idx_sorted], dtype=np.float64)
        self.X = X

        return self

    def transform(self, X):
        """Project data to the adapted feature space."""
        backend = infer_backend(X)
        x_np = to_numpy(X)
        if not hasattr(self, "X"):
            raise ValueError("The fit method should be called before transform.")

        x_kernel_matrix = pairwise_kernels(x_np, self.X, metric=self.kernel, filter_params=True, **self.kwargs)
        x_transformed = np.dot(x_kernel_matrix, self.U[:, : self.n_components])
        return to_backend(x_transformed, backend, reference=X)

    def fit_transform(self, X, y=None, covariates=None, target_covariate=None):
        """Fit the transformer and return the projected samples."""
        self.fit(X, y=y, covariates=covariates, target_covariate=target_covariate)
        return self.transform(X)
