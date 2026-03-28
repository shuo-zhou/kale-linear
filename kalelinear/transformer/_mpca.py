# =============================================================================
# Authors: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#          Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
#          Lalu Muhammad Riza Rizky, l.m.rizky@sheffield.ac.uk
# =============================================================================

"""Tensor and kernel domain adaptation transformers.

This module provides the MPCA transformer and associated utilities.
"""
import logging
import warnings

import numpy as np
import scipy.linalg as la
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly.base import fold, unfold
from tensorly.tenalg import multi_mode_dot


def _check_n_dim(x, n_dims):
    """Validate the number of dimensions.

    Parameters
    ----------
    x : ndarray of shape (n_samples, I_1, ..., I_N)
        Input tensor data.
    n_dims : int
        Expected number of dimensions.

    Raises
    ------
    ValueError
        If ``x.ndim`` does not match ``n_dims``.
    """
    if not x.ndim == n_dims:
        error_msg = "The expected number of dimensions is %s but it is %s for given data" % (n_dims, x.ndim)
        logging.error(error_msg)
        raise ValueError(error_msg)


def _check_shape(x, shape_):
    """Validate per-sample tensor shape.

    Parameters
    ----------
    x : ndarray of shape (n_samples, I_1, ..., I_N)
        Input tensor data.
    shape_ : tuple
        Expected per-sample shape ``(I_1, ..., I_N)``.

    Raises
    ------
    ValueError
        If the sample shape of ``x`` does not match ``shape_``.
    """
    if not x.shape[1:] == shape_:
        error_msg = "The expected shape of data is %s, but %s for given data" % (x.shape[1:], shape_)
        logging.error(error_msg)
        raise ValueError(error_msg)


def _check_tensor_dim_shape(x, n_dims, shape_):
    """Validate both tensor dimensionality and sample shape.

    Parameters
    ----------
    x : ndarray of shape (n_samples, I_1, ..., I_N)
        Input tensor data.
    n_dims : int
        Expected number of dimensions.
    shape_ : tuple
        Expected per-sample shape.
    """
    _check_n_dim(x, n_dims)
    _check_shape(x, shape_)


class MPCA(BaseEstimator, TransformerMixin):
    """Multilinear Principal Component Analysis (MPCA) estimator.

    Parameters
    ----------
    var_ratio : float, default=0.97
        Target cumulative explained variance ratio per mode.
    max_iter : int, default=1
        Maximum number of alternating optimization iterations.
    vectorize : bool, default=False
        If ``True``, output projected tensors as vectors.
    n_components : int, optional
        Number of output features when ``vectorize=True``.

    Attributes
    ----------
    proj_mats : list of ndarray
        Transposed projection matrices with shapes ``(P_i, I_i)``.
    idx_order : ndarray
        Feature ranking indices by descending projected variance.
    mean_ : ndarray
        Per-feature empirical mean of the training data.
    shape_in : tuple
        Input per-sample tensor shape.
    shape_out : tuple
        Output per-sample tensor shape after projection.

    References
    ----------
        Haiping Lu, K.N. Plataniotis, and A.N. Venetsanopoulos, "MPCA: Multilinear Principal Component Analysis of
        Tensor Objects", IEEE Transactions on Neural Networks, Vol. 19, No. 1, Page: 18-39, January 2008. For initial
        Matlab implementation, please go to https://uk.mathworks.com/matlabcentral/fileexchange/26168.

    Examples
    --------
        >>> import numpy as np
        >>> from kalelinear.transformer import MPCA
        >>> x = np.random.random((40, 20, 25, 20))
        >>> x.shape
        (40, 20, 25, 20)
        >>> mpca = MPCA()
        >>> x_projected = mpca.fit_transform(x)
        >>> x_projected.shape
        (40, 18, 23, 18)
        >>> x_projected = mpca.transform(x)
        >>> x_projected.shape
        (40, 7452)
        >>> x_projected = mpca.transform(x)
        >>> x_projected.shape
        (40, 50)
        >>> x_rec = mpca.inverse_transform(x_projected)
        >>> x_rec.shape
        (40, 20, 25, 20)
    """

    def __init__(self, var_ratio=0.97, max_iter=1, vectorize=False, n_components=None):
        self.var_ratio = var_ratio
        if max_iter > 0 and isinstance(max_iter, int):
            self.max_iter = max_iter
        else:
            msg = "Number of max iterations must be a positive integer but given %s" % max_iter
            logging.error(msg)
            raise ValueError(msg)
        self.proj_mats = []
        self.vectorize = vectorize
        self.n_components = n_components

    def fit(self, x, y=None):
        """Fit MPCA to tensor data.

        Parameters
        ----------
        x : ndarray of shape (n_samples, I_1, ..., I_N)
            Input tensor samples.
        y : None, default=None
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : MPCA
            Fitted estimator.
        """
        self._fit(x)
        return self

    def _fit(self, x):
        """Internal solver for MPCA projection matrices.

        Parameters
        ----------
        x : ndarray of shape (n_samples, I_1, ..., I_N)
            Input tensor samples.

        Returns
        -------
        self : MPCA
            Fitted estimator.
        """

        shape_ = x.shape  # shape of input data
        n_dims = x.ndim

        self.shape_in = shape_[1:]
        self.mean_ = np.mean(x, axis=0)
        x = x - self.mean_

        # init
        shape_out = ()
        proj_mats = []

        # get the output tensor shape based on the cumulative distribution of eigen values for each mode
        for i in range(1, n_dims):
            mode_data_mat = unfold(x, mode=i)
            singular_vec_left, singular_val, singular_vec_right = la.svd(mode_data_mat, full_matrices=False)
            eig_values = np.square(singular_val)
            idx_sorted = (-1 * eig_values).argsort()
            cum = eig_values[idx_sorted]
            tot_var = np.sum(cum)

            for j in range(1, cum.shape[0] + 1):
                if np.sum(cum[:j]) / tot_var > self.var_ratio:
                    shape_out += (j,)
                    break
            proj_mats.append(singular_vec_left[:, idx_sorted][:, : shape_out[i - 1]].T)

        # set n_components to the maximum n_features if it is None
        if self.n_components is None:
            self.n_components = int(np.prod(shape_out))

        for i_iter in range(self.max_iter):
            for i in range(1, n_dims):  # ith mode
                x_projected = multi_mode_dot(
                    x,
                    [proj_mats[m] for m in range(n_dims - 1) if m != i - 1],
                    modes=[m for m in range(1, n_dims) if m != i],
                )
                mode_data_mat = unfold(x_projected, i)

                singular_vec_left, singular_val, singular_vec_right = la.svd(mode_data_mat, full_matrices=False)
                eig_values = np.square(singular_val)
                idx_sorted = (-1 * eig_values).argsort()
                proj_mats[i - 1] = (singular_vec_left[:, idx_sorted][:, : shape_out[i - 1]]).T

        x_projected = multi_mode_dot(x, proj_mats, modes=[m for m in range(1, n_dims)])
        x_proj_unfold = unfold(x_projected, mode=0)  # unfold the tensor projection to shape (n_samples, n_features)
        # x_proj_cov = np.diag(np.dot(x_proj_unfold.T, x_proj_unfold))  # covariance of unfolded features
        x_proj_cov = np.sum(np.multiply(x_proj_unfold.T, x_proj_unfold.T), axis=1)  # memory saving computing covariance
        idx_order = (-1 * x_proj_cov).argsort()

        self.proj_mats = proj_mats
        self.idx_order = idx_order
        self.shape_out = shape_out
        self.n_dims = n_dims

        return self

    def transform(self, x):
        """Project data to the MPCA subspace.

        Parameters
        ----------
        x : ndarray of shape (n_samples, I_1, ..., I_N) or (I_1, ..., I_N)
            Input tensor data.

        Returns
        -------
        x_projected : ndarray
            Projected data. Shape is ``(n_samples, P_1, ..., P_N)`` when
            ``vectorize=False``. Otherwise returns vectorized features with
            optional truncation to ``n_components``.
        """
        # reshape x to shape (1, I_1, I_2, ..., I_N) if x in shape (I_1, I_2, ..., I_N), i.e. n_samples = 1
        if x.ndim == self.n_dims - 1:
            x = x.reshape((1,) + x.shape)
        _check_tensor_dim_shape(x, self.n_dims, self.shape_in)
        x = x - self.mean_

        # projected tensor in lower dimensions
        x_projected = multi_mode_dot(x, self.proj_mats, modes=[m for m in range(1, self.n_dims)])

        if self.vectorize:
            x_projected = unfold(x_projected, mode=0)
            x_projected = x_projected[:, self.idx_order]
            if isinstance(self.n_components, int):
                n_features = int(np.prod(self.shape_out))
                if self.n_components > n_features:
                    self.n_components = n_features
                    warn_msg = "n_components exceeds the maximum number, all features will be returned."
                    logging.warning(warn_msg)
                    warnings.warn(warn_msg)
                x_projected = x_projected[:, : self.n_components]

        return x_projected

    def inverse_transform(self, x):
        """Reconstruct original-space tensors from projected data.

        Parameters
        ----------
        x : ndarray
            Projected tensor data, either in tensor or vectorized format.

        Returns
        -------
        x_rec : ndarray of shape (n_samples, I_1, ..., I_N)
            Reconstructed tensor data in the original shape.
        """
        # reshape x to tensor in shape (n_samples, self.shape_out) if x has been unfolded
        if x.ndim <= 2:
            if x.ndim == 1:
                # reshape x to a 2D matrix (1, n_components) if x in shape (n_components,)
                x = x.reshape((1, -1))
            n_samples = x.shape[0]
            n_features = x.shape[1]
            if n_features <= np.prod(self.shape_out):
                x_ = np.zeros((n_samples, np.prod(self.shape_out)))
                x_[:, self.idx_order[:n_features]] = x[:]
            else:
                msg = "Feature dimension exceeds the shape upper limit."
                logging.error(msg)
                raise ValueError(msg)

            x = fold(x_, mode=0, shape=((n_samples,) + self.shape_out))

        x_rec = multi_mode_dot(x, self.proj_mats, modes=[m for m in range(1, self.n_dims)], transpose=True)

        x_rec = x_rec + self.mean_

        return x_rec
