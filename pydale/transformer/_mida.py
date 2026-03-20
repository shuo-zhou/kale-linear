# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================

import numpy as np
from numpy.linalg import multi_dot

from ..utils import base_init
from .base import BaseTransformer


class MIDA(BaseTransformer):
    def __init__(
        self, n_components, penalty=None, kernel="linear", lambda_=1.0, mu=1.0, eta=1.0, augmentation=True, **kwargs
    ):
        """Maximum independence domain adaptation

        Parameters
        ----------
        n_components : int
            n_components after tca (n_components <= d)
        kernel : str
            'rbf' | 'linear' | 'poly' (default is 'linear')
        penalty : str
            None | 'l2' (default is None)
        lambda_ : float
            regularisation param (if penalty==l2)
        mu: total captured variance param
        eta: label dependence param

        References
        ----------
        Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace
        using domain features and independence maximization. IEEE transactions on
        cybernetics, 48(1), pp.288-299.
        """
        super().__init__(n_components, kernel, **kwargs)
        self.lambda_ = lambda_
        self.penalty = penalty
        self.mu = mu
        self.eta = eta
        self.augmentation = augmentation

    def fit(self, x, y=None, covariates=None):
        """
        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)
        y : array-like
            Labels, shape (nl_samples,)
        covariates : array-like
            Domain co-variates, shape (n_samples, n_co-variates)

        Note
        ----
            Unsupervised MIDA is performed if ys and yt are not given.
            Semi-supervised MIDA is performed is ys and yt are given.
        """
        if self.augmentation and type(covariates) == np.ndarray:
            x = np.concatenate((x, covariates), axis=1)
        krnl_x, unit_mat, ctr_mat, n = base_init(x, kernel=self.kernel, **self.kwargs)
        krnl_x = self._centerer.fit_transform(krnl_x)
        if type(covariates) == np.ndarray:
            ker_c = np.dot(covariates, covariates.T)
        else:
            ker_c = np.zeros((n, n))
        if y is not None:
            y_mat = self._lb.fit_transform(y)
            ker_y = np.dot(y_mat, y_mat.T)
            obj = multi_dot([krnl_x, ctr_mat, ker_c, ctr_mat, krnl_x.T])
            st = multi_dot(
                [krnl_x, ctr_mat, (self.mu * unit_mat + self.eta * ker_y / np.square(n - 1)), ctr_mat, krnl_x.T,]
            )
        # obj = np.trace(np.dot(K,L))
        else:
            obj = multi_dot([krnl_x, ctr_mat, ker_c, ctr_mat, krnl_x.T]) / np.square(n - 1)
            st = multi_dot([krnl_x, ctr_mat, krnl_x.T])

        # if self.penalty == 'l2':
        #     obj -= self.lambda_ * unit_mat
        self._fit(obj_min=obj, obj_max=st)
        # self.components_ = np.dot(X.T, U)
        # self.components_ = self.components_.T

        self.x_fit = x
        return self

    def fit_transform(self, x, y=None, covariates=None):
        """
        Parameters
        ----------
        x : array-like,
            shape (n_samples, n_features)
        y : array-like
            shape (n_samples,)
        covariates : array-like
            shape (n_samples, n_co-variates)

        Returns
        -------
        array-like
            transformed X_transformed
        """
        self.fit(x, y, covariates)

        return self.transform(x, covariates)
