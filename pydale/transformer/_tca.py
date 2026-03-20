# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
import numpy as np
from numpy.linalg import multi_dot

from ..utils import base_init, lap_norm, mmd_coef
from .base import BaseTransformer


class TCA(BaseTransformer):
    def __init__(self, n_components, kernel="linear", lambda_=1.0, mu=1.0, gamma_=0.5, k_neighbour=3, **kwargs):
        """Transfer Component Analysis: TCA

        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= (N, d))
        kernel: str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        lambda_ : float
            regularisation param
        mu : float
            KNN graph param
        k_neighbour : int
            number of nearest neighbour for KNN graph
        gamma : float
            label dependence param

        References
        ----------
        S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via Transfer Component Analysis,"
        IEEE Transactions on Neural Networks, 22(2), 199-210, Feb. 2011.
        """
        super().__init__(n_components, kernel, **kwargs)
        self.lambda_ = lambda_
        self.mu = mu
        self.gamma_ = gamma_
        self.k_neighbour = k_neighbour

    def fit(self, xs, ys=None, xt=None, yt=None):
        """[summary]
            Unsupervised TCA is performed if ys and yt are not given.
            Semi-supervised TCA is performed is ys and yt are given.

        Parameters
        ----------
        xs : array-like
            Source domain data, shape (ns_samples, n_features)
        xt : array-like
            Target domain data, shape (nt_samples, n_features)
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None
        """
        if type(xt) == np.ndarray:
            x = np.vstack((xs, xt))
            ns = xs.shape[0]
            nt = xt.shape[0]
            L = mmd_coef(ns, nt, kind="marginal", mu=0)
            L[np.isnan(L)] = 0
        else:
            x = xs.copy()
            L = np.zeros((x.shape[0], x.shape[0]))

        krnl_x, unit_mat, ctr_mat, n = base_init(x, kernel=self.kernel, **self.kwargs)
        krnl_x = self._centerer.fit_transform(krnl_x)

        obj = self.lambda_ * unit_mat
        st = multi_dot([krnl_x, ctr_mat, krnl_x.T])
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
            lap_mat = lap_norm(x, n_neighbour=self.k_neighbour, mode="connectivity")
            obj += multi_dot([krnl_x, (L + self.mu * lap_mat), krnl_x.T])
            st += multi_dot([krnl_x, ctr_mat, ker_y, ctr_mat, krnl_x.T])
        # obj = np.trace(np.dot(krnl_x,L))
        else:
            obj += multi_dot([krnl_x, L, krnl_x.T])

        # obj_ovr = np.dot(inv(obj), st)
        # eig_values, eig_vectors = eig(obj_ovr)
        # idx_sorted = eig_values.argsort()[::-1]
        #
        # self.U = np.asarray(eig_vectors[:, idx_sorted], dtype=np.float)
        # self.x_fit = np.vstack((xs, xt))

        self._fit(obj_min=obj, obj_max=st)
        self.x_fit = x

        return self

    def fit_transform(self, xs, ys=None, xt=None, yt=None):
        """
        Parameters
        ----------
        xs : array-like
            Source domain data, shape (ns_samples, n_features).
        ys : array-like, optional
            Source domain labels, shape (ns_samples,), by default None.
        xt : array-like
            Target domain data, shape (nt_samples, n_features), by default None.
        yt : array-like, optional
            Target domain labels, shape (nt_samples,), by default None.

        Returns
        -------
        array-like
            transformed Xs_transformed, Xt_transformed
        """
        self.fit(xs, ys, xt, yt)

        return self.transform(xs), self.transform(xt)
