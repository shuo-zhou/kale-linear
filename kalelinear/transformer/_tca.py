# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
import numpy as np
from numpy.linalg import multi_dot
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer

# from sklearn.utils.validation import check_is_fitted
from ..utils import base_init, infer_backend, lap_norm, mmd_coef, to_backend, to_numpy


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
            regulisation param
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

    def fit(self, Xs, ys=None, Xt=None, yt=None):
        """[summary]
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.

        Parameters
        ----------
        Xs : array-like
            Source domain data, shape (ns_samples, n_features)
        Xt : array-like
            Target domain data, shape (nt_samples, n_features)
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None
        """
        self.backend_ = infer_backend(Xs, ys, Xt, yt)
        Xs = to_numpy(Xs)
        ys = to_numpy(ys)
        Xt = to_numpy(Xt)
        yt = to_numpy(yt)
        if type(Xt) is np.ndarray:
            X = np.vstack((Xs, Xt))
            ns = Xs.shape[0]
            nt = Xt.shape[0]
            L = mmd_coef(ns, nt, kind="marginal", mu=0)
            L[np.isnan(L)] = 0
        else:
            X = Xs
            L = np.zeros((X.shape[0], X.shape[0]))

        ker_x, unit_mat, ctr_mat, n = base_init(X, kernel=self.kernel, **self.kwargs)

        obj = self.lambda_ * unit_mat
        st = multi_dot([ker_x, ctr_mat, ker_x.T])
        if ys is not None:
            # semisupervised TCA (SSTCA)
            ys_mat = self._lb.fit_transform(ys)
            n_class = ys_mat.shape[1]
            y = np.zeros((n, n_class))
            y[: ys_mat.shape[0], :] = ys_mat[:]
            if yt is not None:
                yt_mat = self._lb.transform(yt)
                y[ys_mat.shape[0] : yt_mat.shape[0], :] = yt_mat[:]
            ker_y = self.gamma_ * np.dot(y, y.T) + (1 - self.gamma_) * unit_mat
            lap_mat = lap_norm(X, n_neighbour=self.k, mode="connectivity")
            obj += multi_dot([ker_x, (L + self.mu * lap_mat), ker_x.T])
            st += multi_dot([ker_x, ctr_mat, ker_y, ctr_mat, ker_x.T])
        # obj = np.trace(np.dot(ker_x,L))
        else:
            obj += multi_dot([ker_x, L, ker_x.T])

        eig_values, eig_vectors = eig(obj, st)
        idx_sorted = eig_values.argsort()

        self.U = np.asarray(eig_vectors[:, idx_sorted], dtype=np.float64)
        self.Xs = Xs
        self.Xt = Xt

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
        X_np = to_numpy(X)
        X_fit = np.vstack((self.Xs, self.Xt))
        ker_x = pairwise_kernels(X_np, X_fit, metric=self.kernel, filter_params=True, **self.kwargs)

        X_transformed = np.dot(ker_x, self.U[:, : self.n_components])
        return to_backend(X_transformed, backend, reference=X)

    def fit_transform(self, Xs, ys=None, Xt=None, yt=None):
        """
        Parameters
        ----------
        Xs : array-like
            Source domain data, shape (ns_samples, n_features).
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None.
        Xt : array-like
            Target domain data, shape (nt_samples, n_features), by default None.
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None.

        Returns
        -------
        array-like
            transformed Xs_transformed, Xt_transformed
        """
        self.fit(Xs, ys, Xt, yt)

        return self.transform(Xs), self.transform(Xt)
