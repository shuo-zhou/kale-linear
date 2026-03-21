# =============================================================================
# author: Shuo Zhou, The University of Sheffield
# =============================================================================
import numpy as np
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels

from ..utils import base_init, mmd_coef

# from sklearn.preprocessing import StandardScaler
# =============================================================================
# Implementation of three transfer learning methods:
#   1. Transfer Component Analysis: TCA
#   2. Joint Distribution Adaptation: JDA
#   3. Balanced Distribution Adaptation: BDA
# Ref:
# [1] S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via
# Transfer Component Analysis," in IEEE Transactions on Neural Networks,
# vol. 22, no. 2, pp. 199-210, Feb. 2011.
# [2] Mingsheng Long, Jianmin Wang, Guiguang Ding, Jiaguang Sun, Philip S. Yu,
# Transfer Feature Learning with Joint Distribution Adaptation, IEEE
# International Conference on Computer Vision (ICCV), 2013.
# [3] Wang, J., Chen, Y., Hao, S., Feng, W. and Shen, Z., 2017, November. Balanced
# distribution adaptation for transfer learning. In Data Mining (ICDM), 2017
# IEEE International Conference on (pp. 1129-1134). IEEE.
# =============================================================================


class JDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel="linear", lambda_=1.0, mu=1.0, **kwargs):
        """
        Parameters
            n_components: n_components after (n_components <= min(d, n))
            kernel_type: [‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’,
            ‘cosine’] (default is 'linear')
            **kwargs: kernel param
            lambda_: regulisation param
            mu: >= 0, param for conditional mmd, (mu=0 for TCA, mu=1 for JDA, BDA otherwise)
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.kernel = kernel
        self.lambda_ = lambda_
        self.mu = mu

    def fit(self, Xs, ys=None, Xt=None, yt=None):
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
        """
        if type(Xt) is np.ndarray:
            X = np.vstack((Xs, Xt))
            ns = Xs.shape[0]
            nt = Xt.shape[0]

            if ys is not None and yt is not None:
                L = mmd_coef(ns, nt, ys, yt, kind="joint", mu=self.mu)
            else:
                L = mmd_coef(ns, nt, kind="marginal", mu=0)
        else:
            X = Xs
            L = np.zeros((X.shape[0], X.shape[0]))

        ker_x, unit_mat, ctr_mat, n = base_init(X, kernel=self.kernel, **self.kwargs)

        # objective for optimization
        obj = np.dot(np.dot(ker_x, L), ker_x.T) + self.lambda_ * unit_mat
        # constraint subject to
        st = np.dot(np.dot(ker_x, ctr_mat), ker_x.T)
        eig_values, eig_vectors = eig(obj, st)

        ev_abs = np.array(list(map(lambda item: np.abs(item), eig_values)))
        #        idx_sorted = np.argsort(ev_abs)[:self.n_components]
        idx_sorted = np.argsort(ev_abs)

        U = np.zeros(eig_vectors.shape)
        U[:, :] = eig_vectors[:, idx_sorted]
        self.U = np.asarray(U, dtype=np.float64)
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
        # X = self.scaler.transform(X)
        # check_is_fitted(self, 'Xs')
        # check_is_fitted(self, 'Xt')
        X_fit = np.vstack((self.Xs, self.Xt))
        ker_x = pairwise_kernels(X, X_fit, metric=self.kernel, filter_params=True, **self.kwargs)

        return np.dot(ker_x, self.U[:, : self.n_components])

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
