# =============================================================================
# @author: Shuo Zhou, Lalu Muhammad Riza Rizky, The University of Sheffield
# @contact: shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Base classes and utilities for domain adaptation methods.

This module provides shared utilities and base classes for kernel-based domain adaptation
methods such as TCA, JDA, and MIDA.

Available utilities:
  - compute_eigendecomposition: Perform generalized eigendecomposition
  - get_kernel: Compute kernel matrices
  - BaseKernelDomainAdapter: Base class for kernel-based methods

Usage:
  Other domain adaptation methods can import and use these utilities:

  from kalelinear.transformer._base import compute_eigendecomposition, get_kernel

  or inherit from BaseKernelDomainAdapter for method-level shared functionality.
"""

from numbers import Integral, Real

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
from sklearn.base import _fit_context, BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.preprocessing import FunctionTransformer, KernelCenterer, OneHotEncoder
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import _randomized_eigsh, safe_sparse_dot, svd_flip
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    _check_psd_eigenvalues,
    _num_features,
    _num_samples,
    check_is_fitted,
    NotFittedError,
    validate_data,
)

# from kalelinear.utils import base_init, infer_backend, to_backend, to_numpy


def _centering_kernel(size, dtype=np.float64):
    """Generate a centering matrix.

    Parameters
    ----------
    size : int
        Number of rows/columns of the square matrix.
    dtype : data-type, default=np.float64
        Output dtype.

    Returns
    -------
    ndarray of shape (size, size)
        Centering matrix ``I - 1/n``.
    """

    identity = np.eye(size, dtype=dtype)
    ones_matrix = np.ones((size, size), dtype)
    return identity - ones_matrix / size


def _check_num_components(k, num_components):
    """Resolve the effective number of components.

    Parameters
    ----------
    k : ndarray
        Kernel matrix.
    num_components : int or None
        Requested number of components.

    Returns
    -------
    int
        Effective number of components.
    """
    k_size = _num_features(k)
    if num_components is None:
        num_components = k_size  # use all dimensions
    else:
        num_components = min(k_size, num_components)
    return num_components


# solver helper


def _check_solver(k, num_components, solver):
    """Resolve eigensolver strategy.

    Parameters
    ----------
    k : ndarray
        Kernel matrix.
    num_components : int
        Number of components to keep.
    solver : {"auto", "arpack", "randomized", "dense"}
        Requested solver.

    Returns
    -------
    str
        Effective solver name.
    """
    k_size = _num_features(k)

    if solver == "auto" and k_size > 200 and num_components < 10:
        solver = "arpack"
    elif solver == "auto":
        solver = "dense"
    return solver


def _eigendecompose(
    k,
    num_components=None,
    solver="auto",
    random_state=None,
    max_iter=None,
    tol=0,
    iterated_power="auto",
):
    """Compute eigenpairs for a kernel matrix.

    Parameters
    ----------
    k : ndarray or tuple of ndarray
        Kernel matrix ``a`` or generalized pair ``(a, b)``.
    num_components : int, optional
        Number of components to keep.
    solver : {"auto", "arpack", "randomized", "dense"}, default="auto"
        Eigensolver backend.
    random_state : int, RandomState instance or None, default=None
        Random state used by stochastic solver.
    max_iter : int, optional
        Maximum iterations for iterative solver.
    tol : float, default=0
        Convergence tolerance.
    iterated_power : int or {"auto"}, default="auto"
        Power iterations for randomized solver.

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues.
    eigenvectors : ndarray
        Corresponding eigenvectors.
    """
    # we accept tuple or list for k, in case a method
    # need to use a generalized eigenvalue decomposition
    if isinstance(k, (tuple, list)):
        a, b = k
    else:
        a, b = k, None

    num_components = _check_num_components(k, num_components)
    solver = _check_solver(k, num_components, solver)

    if solver == "arpack":
        v0 = _init_arpack_v0(_num_features(a), random_state)
        return eigsh(
            a,
            num_components,
            b,
            which="LA",
            tol=tol,
            maxiter=max_iter,
            v0=v0,
        )

    if solver == "randomized":
        # To support methods that require generalized eigendecomposition,
        # for randomized solver that doesn't support it by default.
        # We use the inverse of b to transform a to obtain an equivalent
        # formulation using regular eigendecomposition.
        if b is not None:
            a = la.inv(b) @ a

        return _randomized_eigsh(
            a,
            n_components=num_components,
            n_iter=iterated_power,
            random_state=random_state,
            selection="module",
        )

    # If solver is 'dense', use standard scipy.linalg.eigh
    # Note: subset_by_index specifies the indices of smallest/largest to return
    index = (_num_features(a) - num_components, _num_features(a) - 1)
    return la.eigh(a, b, subset_by_index=index)


# Postprocess eignevalues and eigenvectors


def _postprocess_eigencomponents(eigenvalues, eigenvectors, steps, num_components=None, remove_zero_eig=False):
    """Postprocess eigenvalues and eigenvectors.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues from decomposition.
    eigenvectors : ndarray
        Eigenvectors from decomposition.
    steps : list of str
        Processing steps to apply in order.
    num_components : int, optional
        Requested number of output components.
    remove_zero_eig : bool, default=False
        Whether to drop zero-valued eigencomponents.

    Returns
    -------
    eigenvalues : ndarray
        Processed eigenvalues.
    eigenvectors : ndarray
        Processed eigenvectors.
    """
    for step in steps:
        if step == "remove_significant_negative_eigenvalues":
            eigenvalues = _remove_significant_negative_eigenvalues(eigenvalues)
        if step == "check_psd_eigenvalues":
            eigenvalues = _check_psd_eigenvalues(eigenvalues)
        if step == "svd_flip":
            eigenvectors, _ = svd_flip(eigenvectors, None)
        if step == "sort_eigencomponents":
            eigenvalues, eigenvectors = _sort_eigencomponents(eigenvalues, eigenvectors)
        if step == "keep_positive_eigenvalues":
            eigenvalues, eigenvectors = _keep_positive_eigenvalues(
                eigenvalues, eigenvectors, num_components, remove_zero_eig
            )

    return eigenvalues, eigenvectors


def _sort_eigencomponents(eigenvalues, eigenvectors):
    """Sort eigencomponents by descending eigenvalue.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues.
    eigenvectors : ndarray
        Eigenvectors.

    Returns
    -------
    eigenvalues : ndarray
        Sorted eigenvalues.
    eigenvectors : ndarray
        Sorted eigenvectors.
    """
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    return eigenvalues, eigenvectors


def _keep_positive_eigenvalues(eigenvalues, eigenvectors, num_components=None, remove_zero_eig=False):
    """Filter non-positive eigencomponents.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues.
    eigenvectors : ndarray
        Eigenvectors.
    num_components : int, optional
        Requested number of components.
    remove_zero_eig : bool, default=False
        Whether to remove zero eigenvalues.

    Returns
    -------
    eigenvalues : ndarray
        Filtered eigenvalues.
    eigenvectors : ndarray
        Filtered eigenvectors.
    """
    if remove_zero_eig or num_components is None:
        pos_mask = eigenvalues > 0
        eigenvectors = eigenvectors[:, pos_mask]
        eigenvalues = eigenvalues[pos_mask]

    return eigenvalues, eigenvectors


def _remove_significant_negative_eigenvalues(lambdas):
    """Clip significant negative eigenvalues to zero.

    Parameters
    ----------
    lambdas : array-like
        Eigenvalues.

    Returns
    -------
    ndarray
        Eigenvalues with significant negative entries removed.
    """
    lambdas = np.array(lambdas)
    is_double_precision = lambdas.dtype == np.float64
    significant_neg_ratio = 1e-5 if is_double_precision else 5e-3
    significant_neg_value = 1e-10 if is_double_precision else 1e-6

    lambdas = np.real(lambdas)
    max_eig = lambdas.max()

    significant_neg_eigvals_index = lambdas < -significant_neg_ratio * max_eig
    significant_neg_eigvals_index &= lambdas < -significant_neg_value

    lambdas[significant_neg_eigvals_index] = 0

    return lambdas


def _scale_eigenvectors(eigenvalues, eigenvectors):
    """Scale eigenvectors by the square root of eigenvalues.

    Parameters
    ----------
    eigenvalues : ndarray
        Eigenvalues.
    eigenvectors : ndarray
        Eigenvectors.

    Returns
    -------
    ndarray
        Scaled eigenvectors.
    """
    s = np.sqrt(eigenvalues)

    non_zeros = np.flatnonzero(s)
    eigenvectors_ = np.zeros_like(eigenvectors)
    eigenvectors_[:, non_zeros] = eigenvectors[:, non_zeros] / s[non_zeros]

    return eigenvectors_


class BaseKernelDomainAdapter(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Base class for kernel domain adaptation methods. Extendable to support different
    kernel-based domain adaptation methods (e.g., MIDA, TCA, SCA).

    Parameters
    ----------
        num_components (int, optional): Number of components to keep. If None, all components are kept. Defaults to None.
        ignore_y (bool, optional): Whether to ignore the target variable `y` during fitting. Defaults to False.
        augment (str, optional): Whether to augment the input data with factors. Can be "pre" (prepend factors),
            "post" (append factors), or None (no augmentation). Defaults to None.
        kernel (str or callable, optional): Kernel function to use. Can be "linear", "rbf", "poly", "sigmoid", or a callable. Defaults to "linear".
        gamma (float, optional): Kernel coefficient for "rbf", "poly", and "sigmoid" kernels. If None, defaults to 1 / num_features. Defaults to None.
        degree (int, optional): Degree of the polynomial kernel. Ignored by other kernels. Defaults to 3.
        coef0 (float, optional): Independent term in the polynomial and sigmoid kernels. Ignored by other kernels. Defaults to 1.
        kernel_params (dict, optional): Additional parameters for the kernel function. Defaults to None.
        alpha (float, optional): Regularization parameter for the kernel. Defaults to 1.0.
        fit_inverse_transform (bool, optional): Whether to fit the inverse transform for reconstruction. Defaults to False.
        eigen_solver (str, optional): Eigenvalue solver to use. Can be "auto", "dense", "arpack", or "randomized". Defaults to "auto".
        tol (float, optional): Tolerance for convergence of the eigenvalue solver. Defaults to 0.
        max_iter (int, optional): Maximum number of iterations for the eigenvalue solver. If None, no limit is applied. Defaults to None.
        iterated_power (int or str, optional): Number of iterations for the randomized solver. Can be an integer or "auto". Defaults to "auto".
        remove_zero_eig (bool, optional): Whether to remove zero eigenvalues during postprocessing. Defaults to False.
        scale_components (bool, optional): Whether to scale the components by the square root of their eigenvalues. Defaults to False.
        random_state (int, np.random.RandomState, or None, optional): Random seed for reproducibility. Defaults to None.
        copy (bool, optional): Whether to copy the input data during validation. Defaults to True.
        num_jobs (int or None, optional): Number of jobs to run in parallel for pairwise kernel computations. Defaults to None.
    """

    _parameter_constraints: dict = {
        "num_components": [Interval(Integral, 1, None, closed="left"), None],
        "ignore_y": ["boolean"],
        "augment": [StrOptions({"pre", "post"}), None],
        "kernel": [
            StrOptions(set(PAIRWISE_KERNEL_FUNCTIONS) | {"precomputed"}),
            callable,
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "alpha": [Interval(Real, 0, None, closed="left")],
        "fit_inverse_transform": ["boolean"],
        "eigen_solver": [StrOptions({"auto", "dense", "arpack", "randomized"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "iterated_power": [
            Interval(Integral, 0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "remove_zero_eig": ["boolean"],
        "scale_components": ["boolean"],
        "random_state": ["random_state"],
        "copy": ["boolean"],
        "num_jobs": [None, Integral],
    }

    _eigen_preprocess_steps = [
        "remove_significant_negative_eigenvalues",
        "check_psd_eigenvalues",
        "svd_flip",
        "sort_eigencomponents",
        "keep_positive_eigenvalues",
    ]

    def __init__(
        self,
        num_components=None,
        ignore_y=False,
        augment=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        alpha=1.0,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        num_jobs=None,
    ):
        # Truncation parameters
        self.num_components = num_components

        # Supervision parameters
        self.ignore_y = ignore_y

        # Kernel parameters
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.copy = copy
        self.num_jobs = num_jobs

        # Eigendecomposition parameters
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state

        # Transform parameters
        self.scale_components = scale_components

        # Inverse transform parameters
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform

        # Additional adaptation parameters
        self.augment = augment

    def _fit_inverse_transform(self, x_transformed, x):
        if hasattr(x, "tocsr"):
            raise NotImplementedError("Inverse transform not implemented for sparse matrices!")

        n_samples = x_transformed.shape[0]
        k_x = self._get_kernel(x_transformed)
        k_x.flat[:: n_samples + 1] += self.alpha
        self.dual_coef_ = la.solve(k_x, x, assume_a="pos", overwrite_a=True)
        self.x_transformed_fit_ = x_transformed

    @property
    def _n_features_out(self):
        """Number of features out after transformation."""

        # The property name can't be changed as it is used
        # by scikit-learn's core module to validate.
        return self.eigenvalues_.shape[0]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        tags.input_tags.pairwise = self.kernel == "precomputed"
        return tags

    def _more_tags(self):
        return {
            "_xfail_checks": {"check_transformer_n_iter": "Follows similar implementation to KernelPCA."},
        }

    def _get_kernel(self, x, y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}
        return pairwise_kernels(x, y, metric=self.kernel, filter_params=True, n_jobs=self.num_jobs, **params)

    @property
    def orig_coef_(self):
        """Coefficients projected to the original feature space
        with shape (num_components, num_features).
        """
        check_is_fitted(self)
        if self.kernel != "linear":
            raise NotImplementedError("Available only when `kernel=True`.")
        w = self.eigenvectors_
        if self.scale_components:
            w = _scale_eigenvectors(self.eigenvalues_, w)
        return safe_sparse_dot(w.T, self.x_fit_)

    def _fit_transform_in_place(self, k_x):
        """Fit eigendecomposition on a precomputed kernel matrix.

        Parameters
        ----------
        k_x : ndarray
            Kernel matrix used for eigendecomposition.
        """
        eigenvalues, eigenvectors = _eigendecompose(
            k_x,
            self.num_components,
            self.eigen_solver,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            iterated_power=self.iterated_power,
        )

        eigenvalues, eigenvectors = _postprocess_eigencomponents(
            eigenvalues,
            eigenvectors,
            self._eigen_preprocess_steps,
            num_components=self.num_components,
            remove_zero_eig=self.remove_zero_eig,
        )

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

    def _make_objective_kernel(self, k_x, y, factors):
        """Create the objective kernel for eigendecomposition.

        Parameters
        ----------
        k_x : ndarray
            Centered kernel matrix.
        y : ndarray of shape (n_samples, n_classes)
            Encoded labels.
        factors : ndarray of shape (n_samples, n_factors)
            Preprocessed adaptation factors.

        Returns
        -------
        ndarray
            Objective kernel.
        """
        return k_x

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, x, y=None, group_labels=None, **fit_params):
        """Fit the domain adapter.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), optional
            Target labels. Use ``-1`` for unknown labels in semi-supervised settings.
        group_labels : array-like of shape (n_samples, n_factors), optional
            Preprocessed grouping/domain factors.
        **fit_params : dict
            Additional fit parameters for compatibility.

        Returns
        -------
        self : BaseKernelDomainAdapter
            Fitted estimator.
        """

        # Data validation for x, y, and factors
        check_params = {
            "accept_sparse": False if self.fit_inverse_transform else "csr",
            "copy": self.copy,
        }
        if y is None or self.ignore_y:
            x = validate_data(self, x, **check_params)
            y_ohe = np.zeros((_num_samples(x), 1), dtype=x.dtype)
        else:
            x, y = validate_data(self, x, y, **check_params)
            y_type = type_of_target(y)

            if y_type not in ["binary", "multiclass"]:
                raise ValueError(f"y should be a 'binary' or 'multiclass' target. Got '{y_type}' instead.")

            drop = (-1,) if np.any(y == -1) else None
            ohe = OneHotEncoder(sparse_output=False, drop=drop)
            y_ohe = ohe.fit_transform(np.expand_dims(y, 1))

            self.classes_ = ohe.categories_[0]

        if group_labels is None:
            raise ValueError(f"Group labels must be provided for `{self.__class__.__name__}` during `fit`.")

        # k_objective workaround to validate the group_labels' shape
        factor_validator = FunctionTransformer()
        factor_validator.fit(group_labels, x)
        group_labels = factor_validator.transform(group_labels)

        # Append the factors/phenotypes to the input data if augment=True
        x_aug = x
        if self.augment is not None:
            self._factor_validator = factor_validator

        if self.augment == "pre":
            x_aug = np.hstack((x, group_labels))

        # To avoid having duplicate variables, x_fit_ cannot be renamed
        # to x_fit_ as it is predefined in KernelPCA's implementation
        self.x_fit_ = x_aug
        self.gamma_ = 1 / _num_features(x) if self.gamma is None else self.gamma
        self._centerer = KernelCenterer()

        k_x = self._get_kernel(self.x_fit_)
        k_x = self._centerer.fit_transform(k_x)

        k_objective = self._make_objective_kernel(k_x, y_ohe, group_labels)

        # Fit the transformation and inverse transformation for the kernel matrix
        self._fit_transform_in_place(k_objective)
        if self.fit_inverse_transform:
            x_transformed = self.transform(x, group_labels)
            self._fit_inverse_transform(x_transformed, x)

        return self

    def transform(self, x, group_labels=None):
        """Project data to the adapted feature space.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input data.
        group_labels : array-like of shape (n_samples, n_factors), optional
            Group factors required when augmentation is enabled.

        Returns
        -------
        z : ndarray of shape (n_samples, n_components)
            Projected representation.
        """
        check_is_fitted(self)
        accept_sparse = False if self.fit_inverse_transform else "csr"
        x = validate_data(self, x, accept_sparse=accept_sparse, reset=False)

        if group_labels is None and self.augment in {"pre", "post"}:
            raise ValueError("Factors must be provided for transform when `augment` is 'pre' or 'post'.")

        if self.augment == "pre":
            x = np.hstack((x, group_labels))

        k_x = self._get_kernel(x, self.x_fit_)
        k_x = self._centerer.transform(k_x)

        w = self.eigenvectors_
        if self.scale_components:
            w = _scale_eigenvectors(self.eigenvalues_, w)

        z = safe_sparse_dot(k_x, w)

        if self.augment == "post":
            z = np.hstack((z, group_labels))

        return z

    def inverse_transform(self, z):
        """Map projected features back to the original feature space.

        Parameters
        ----------
        z : array-like of shape (n_samples, n_components)
            Projected features.

        Returns
        -------
        ndarray
            Reconstructed samples in the original feature space.
        """
        check_is_fitted(self)
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        k_z = self._get_kernel(z, self.x_transformed_fit_)
        return safe_sparse_dot(k_z, self.dual_coef_)

    def fit_transform(self, x, y=None, group_labels=None, **fit_params):
        """Fit the adapter and return transformed features.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), optional
            Target labels.
        group_labels : array-like of shape (n_samples, n_factors), optional
            Group/domain factors.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        ndarray
            Projected features.
        """
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            self.fit(x, group_labels=group_labels, **fit_params)
        else:
            # fit method of arity 2 (supervised transformation)
            self.fit(x, y, group_labels, **fit_params)

        return self.transform(x, group_labels)
