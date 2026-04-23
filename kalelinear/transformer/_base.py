# =============================================================================
# @author: Shuo Zhou, Lalu Muhammad Riza Rizky, The University of Sheffield
# @contact: shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Base classes and utilities for kernel domain adaptation methods."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
from sklearn.base import _fit_context, BaseEstimator, ClassNamePrefixFeaturesOutMixin, clone, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels
from sklearn.preprocessing import FunctionTransformer, KernelCenterer, OneHotEncoder
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
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

from kalelinear.utils import mmd_coef


@dataclass
class KernelFitContext:
    """Shared fit-time context for kernel domain adapters."""

    X_input: np.ndarray
    X_fit: np.ndarray
    covariates_input: np.ndarray | None = None
    covariates_fit: np.ndarray | None = None
    y_input: np.ndarray | None = None
    y_fit: np.ndarray | None = None
    y_encoded: np.ndarray | None = None
    classes: np.ndarray | None = None
    source_idx: np.ndarray | None = None
    target_idx: np.ndarray | None = None
    ns: int | None = None
    nt: int | None = None
    ys: np.ndarray | None = None
    yt: np.ndarray | None = None
    target_covariate: object | None = None


def _centering_kernel(size, dtype=np.float64):
    """Generate a centering matrix."""
    identity = np.eye(size, dtype=dtype)
    ones_matrix = np.ones((size, size), dtype)
    return identity - ones_matrix / size


def _get_eigenproblem_matrices(eigenproblem):
    """Normalize an eigenproblem specification to ``(A, B)``."""
    if isinstance(eigenproblem, (tuple, list)):
        return eigenproblem[0], eigenproblem[1]
    return eigenproblem, None


def _check_n_components(eigenproblem, n_components):
    """Resolve the effective number of components."""
    a, _ = _get_eigenproblem_matrices(eigenproblem)
    problem_size = _num_features(a)
    if n_components is None:
        n_components = problem_size
    else:
        n_components = min(problem_size, n_components)
    return n_components


def _check_solver(eigenproblem, n_components, solver, eigenvalue_order):
    """Resolve eigensolver strategy."""
    a, _ = _get_eigenproblem_matrices(eigenproblem)
    problem_size = _num_features(a)

    if solver == "auto" and problem_size > 200 and n_components < 10:
        solver = "arpack"
    elif solver == "auto":
        solver = "dense"

    # Randomized eigensolvers are naturally aligned with dominant eigenpairs.
    if solver == "randomized" and eigenvalue_order == "ascending":
        solver = "dense"

    return solver


def _eigendecompose(
    eigenproblem,
    n_components=None,
    solver="auto",
    random_state=None,
    max_iter=None,
    tol=0,
    iterated_power="auto",
    eigenvalue_order="descending",
):
    """Compute eigenpairs for a kernel matrix or a generalized eigenproblem."""
    a, b = _get_eigenproblem_matrices(eigenproblem)

    n_components = _check_n_components((a, b), n_components)
    solver = _check_solver((a, b), n_components, solver, eigenvalue_order)

    if solver == "arpack":
        v0 = _init_arpack_v0(_num_features(a), random_state)
        which = "LA" if eigenvalue_order == "descending" else "SA"
        return eigsh(
            a,
            n_components,
            b,
            which=which,
            tol=tol,
            maxiter=max_iter,
            v0=v0,
        )

    if solver == "randomized":
        if b is not None:
            a = la.inv(b) @ a

        return _randomized_eigsh(
            a,
            n_components=n_components,
            n_iter=iterated_power,
            random_state=random_state,
            selection="module",
        )

    if eigenvalue_order == "descending":
        index = (_num_features(a) - n_components, _num_features(a) - 1)
    else:
        index = (0, n_components - 1)
    return la.eigh(a, b, subset_by_index=index)


def _postprocess_eigencomponents(
    eigenvalues,
    eigenvectors,
    steps,
    n_components=None,
    remove_zero_eig=False,
    sort_order="descending",
):
    """Postprocess eigenvalues and eigenvectors."""
    for step in steps:
        if step == "remove_significant_negative_eigenvalues":
            eigenvalues = _remove_significant_negative_eigenvalues(eigenvalues)
        if step == "check_psd_eigenvalues":
            eigenvalues = _check_psd_eigenvalues(eigenvalues)
        if step == "svd_flip":
            eigenvectors, _ = svd_flip(eigenvectors, None)
        if step == "sort_eigencomponents":
            eigenvalues, eigenvectors = _sort_eigencomponents(eigenvalues, eigenvectors, sort_order=sort_order)
        if step == "keep_positive_eigenvalues":
            eigenvalues, eigenvectors = _keep_positive_eigenvalues(
                eigenvalues, eigenvectors, n_components, remove_zero_eig
            )

    return eigenvalues, eigenvectors


def _sort_eigencomponents(eigenvalues, eigenvectors, sort_order="descending"):
    """Sort eigencomponents by eigenvalue."""
    indices = eigenvalues.argsort()
    if sort_order == "descending":
        indices = indices[::-1]

    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    return eigenvalues, eigenvectors


def _keep_positive_eigenvalues(eigenvalues, eigenvectors, n_components=None, remove_zero_eig=False):
    """Filter non-positive eigencomponents."""
    if remove_zero_eig or n_components is None:
        pos_mask = eigenvalues > 0
        eigenvectors = eigenvectors[:, pos_mask]
        eigenvalues = eigenvalues[pos_mask]

    return eigenvalues, eigenvectors


def _remove_significant_negative_eigenvalues(lambdas):
    """Clip significant negative eigenvalues to zero."""
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
    """Scale eigenvectors by the square root of eigenvalues."""
    s = np.sqrt(eigenvalues)

    non_zeros = np.flatnonzero(s)
    eigenvectors_ = np.zeros_like(eigenvectors)
    eigenvectors_[:, non_zeros] = eigenvectors[:, non_zeros] / s[non_zeros]

    return eigenvectors_


class BaseKernelDomainAdapter(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Base class for kernel domain adaptation methods.

    Subclasses operate on a feature matrix ``X`` and optional ``covariates``.
    Unless a ``covariate_encoder`` is configured, ``covariates`` must already
    be numeric and in model-ready form. When an encoder is configured, raw
    tabular covariates are accepted during :meth:`fit` and transformed
    consistently during :meth:`transform`.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
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
        "covariate_encoder": [StrOptions({"onehot"}), HasMethods(["fit", "transform"]), None],
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
        "n_jobs": [None, Integral],
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
        n_components=None,
        ignore_y=False,
        augment=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        covariate_encoder=None,
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
        n_jobs=None,
    ):
        self.n_components = n_components
        self.ignore_y = ignore_y
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.covariate_encoder = covariate_encoder
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.copy = copy
        self.n_jobs = n_jobs
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.remove_zero_eig = remove_zero_eig
        self.random_state = random_state
        self.scale_components = scale_components
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.augment = augment

    def _fit_inverse_transform(self, x_transformed, X):
        if hasattr(X, "tocsr"):
            raise NotImplementedError("Inverse transform not implemented for sparse matrices!")

        n_samples = x_transformed.shape[0]
        x_transformed_kernel_matrix = self._get_kernel(x_transformed)
        x_transformed_kernel_matrix.flat[:: n_samples + 1] += self.alpha
        self.dual_coef_ = la.solve(x_transformed_kernel_matrix, X, assume_a="pos", overwrite_a=True)
        self.x_transformed_fit_ = x_transformed

    @property
    def _n_features_out(self):
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

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma_, "degree": self.degree, "coef0": self.coef0}
            if self.kernel_params is not None:
                params.update(self.kernel_params)
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params)

    @property
    def orig_coef_(self):
        check_is_fitted(self)
        if self.kernel != "linear":
            raise NotImplementedError("Available only when `kernel='linear'`.")
        w = self.eigenvectors_
        if self.scale_components:
            w = _scale_eigenvectors(self.eigenvalues_, w)
        return safe_sparse_dot(w.T, self.x_fit_)

    def _requires_covariates(self):
        return False

    def _coerce_covariates_for_encoder(self, covariates):
        covariates = np.asarray(covariates)
        if covariates.ndim == 0 or covariates.ndim > 2:
            raise ValueError("Covariates must be a 1D or 2D array-like object.")
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        return covariates

    def _to_dense_covariates(self, covariates):
        if hasattr(covariates, "toarray"):
            return covariates.toarray()
        return covariates

    def _fit_covariate_encoder(self, covariates):
        if covariates is None:
            self.covariate_encoder_ = None
            return None

        if self.covariate_encoder is None:
            self.covariate_encoder_ = None
            return covariates

        covariates_to_encode = self._coerce_covariates_for_encoder(covariates)
        if self.covariate_encoder == "onehot":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        else:
            encoder = clone(self.covariate_encoder)

        encoder.fit(covariates_to_encode)
        encoded_covariates = encoder.transform(covariates_to_encode)
        self.covariate_encoder_ = encoder
        return self._to_dense_covariates(encoded_covariates)

    def _transform_covariates(self, covariates):
        if covariates is None:
            return None

        encoder = getattr(self, "covariate_encoder_", None)
        if encoder is None:
            return covariates

        covariates_to_encode = self._coerce_covariates_for_encoder(covariates)
        encoded_covariates = encoder.transform(covariates_to_encode)
        return self._to_dense_covariates(encoded_covariates)

    def _validate_covariates(self, covariates, X):
        if covariates is None:
            if self._requires_covariates():
                raise ValueError(f"Covariates must be provided for `{self.__class__.__name__}` during `fit`.")
            return None, None

        covariates = self._coerce_covariates_for_encoder(covariates)
        if covariates.shape[0] != _num_samples(X):
            raise ValueError("Covariates and X must have the same number of samples.")

        covariate_dtype = covariates.dtype
        if not (np.issubdtype(covariate_dtype, np.number) or np.issubdtype(covariate_dtype, np.bool_)):
            raise ValueError(
                "Covariates must be numeric when `covariate_encoder` is None. "
                "Provide numeric covariates or set `covariate_encoder`."
            )

        factor_validator = FunctionTransformer(validate=False)
        factor_validator.fit(covariates, X)
        return factor_validator.transform(covariates), factor_validator

    def _validate_transform_covariates(self, covariates, X):
        if covariates is None:
            return None

        covariates = self._coerce_covariates_for_encoder(covariates)
        if covariates.shape[0] != _num_samples(X):
            raise ValueError("Covariates and X must have the same number of samples.")

        covariate_dtype = covariates.dtype
        if not (np.issubdtype(covariate_dtype, np.number) or np.issubdtype(covariate_dtype, np.bool_)):
            raise ValueError("Transformed covariates must be numeric.")

        if getattr(self, "covariates_fit_", None) is not None:
            fit_covariates = self.covariates_fit_
            expected_n_features = 1 if fit_covariates.ndim == 1 else fit_covariates.shape[1]
            if covariates.shape[1] != expected_n_features:
                raise ValueError(
                    "Covariates during transform must match the fitted covariate feature dimension. "
                    f"Expected {expected_n_features}, got {covariates.shape[1]}."
                )

        return covariates

    def _get_unlabeled_value(self, y):
        if np.issubdtype(np.asarray(y).dtype, np.number):
            return -1
        return "__unlabeled__"

    def _encode_y_for_fit(self, y, n_samples, x_dtype, unlabeled_value=None):
        if y is None or self.ignore_y:
            return None, np.zeros((n_samples, 1), dtype=x_dtype), None

        y = np.asarray(y)
        if _num_samples(y) != n_samples:
            raise ValueError("y and X must have the same number of samples.")

        y_no_unlabeled = y
        has_unlabeled = unlabeled_value is not None and np.any(y == unlabeled_value)
        if has_unlabeled:
            y_no_unlabeled = y[y != unlabeled_value]

        if y_no_unlabeled.size > 0:
            y_type = type_of_target(y_no_unlabeled)
            if y_type not in ["binary", "multiclass"]:
                raise ValueError(f"y should be a 'binary' or 'multiclass' target. Got '{y_type}' instead.")

        drop = [unlabeled_value] if has_unlabeled else None
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop=drop)
        y_encoded = one_hot_encoder.fit_transform(np.expand_dims(y, 1))

        classes = one_hot_encoder.categories_[0]
        if has_unlabeled:
            classes = classes[classes != unlabeled_value]

        return y, y_encoded, classes

    def _prepare_fit_context(self, X, y=None, covariates=None, **fit_params):
        y_fit, y_encoded, classes = self._encode_y_for_fit(y, _num_samples(X), X.dtype)
        return KernelFitContext(
            X_input=X,
            X_fit=X,
            covariates_input=covariates,
            covariates_fit=covariates,
            y_input=y,
            y_fit=y_fit,
            y_encoded=y_encoded,
            classes=classes,
        )

    def _augment_data(self, X, covariates):
        if self.augment == "pre":
            return np.hstack((X, covariates))
        return X

    def _make_eigenproblem(self, x_kernel_matrix, context):
        return x_kernel_matrix

    def _get_eigenvalue_order(self):
        return "descending"

    def _fit_transform_in_place(self, eigenproblem):
        eigenvalues, eigenvectors = _eigendecompose(
            eigenproblem,
            self.n_components,
            self.eigen_solver,
            random_state=self.random_state,
            max_iter=self.max_iter,
            tol=self.tol,
            iterated_power=self.iterated_power,
            eigenvalue_order=self._get_eigenvalue_order(),
        )

        eigenvalues, eigenvectors = _postprocess_eigencomponents(
            eigenvalues,
            eigenvectors,
            self._eigen_preprocess_steps,
            n_components=self.n_components,
            remove_zero_eig=self.remove_zero_eig,
            sort_order=self._get_eigenvalue_order(),
        )

        self.eigenvalues_ = np.real(eigenvalues)
        self.eigenvectors_ = np.real(eigenvectors)
        self.U = np.asarray(self.eigenvectors_, dtype=np.float64)
        self.eig_values_ = np.asarray(self.eigenvalues_, dtype=np.float64)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, covariates=None, **fit_params):
        """Fit the adapter on ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,), default=None
            Optional labels used by label-aware subclasses.
        covariates : array-like, default=None
            Auxiliary covariates aligned with ``X``. When
            ``covariate_encoder is None``, covariates must already be numeric
            and either 1D or 2D. When an encoder is configured, raw 1D or 2D
            tabular covariates are accepted and encoded during fit.
        **fit_params : dict
            Subclass-specific fit parameters.
        """
        check_params = {
            "accept_sparse": False if self.fit_inverse_transform else "csr",
            "copy": self.copy,
        }
        X = validate_data(self, X, **check_params)
        raw_covariates = covariates
        covariates = self._fit_covariate_encoder(covariates)
        covariates, factor_validator = self._validate_covariates(covariates, X)

        context = self._prepare_fit_context(X, y=y, covariates=covariates, **fit_params)
        context.covariates_input = raw_covariates

        if context.classes is not None:
            self.classes_ = context.classes
        elif hasattr(self, "classes_"):
            delattr(self, "classes_")

        if factor_validator is not None and self.augment is not None:
            self._factor_validator = factor_validator
        elif hasattr(self, "_factor_validator"):
            delattr(self, "_factor_validator")

        self.x_fit_ = self._augment_data(context.X_fit, context.covariates_fit)
        self.x_fit_raw_ = context.X_fit
        self.covariates_input_ = raw_covariates
        self.covariates_fit_ = context.covariates_fit
        self.fit_context_ = context
        self.X = context.X_fit

        self.gamma_ = 1 / _num_features(X) if self.gamma is None else self.gamma
        self._centerer = KernelCenterer()

        x_fit_kernel_matrix = self._get_kernel(self.x_fit_)
        x_fit_kernel_matrix = self._centerer.fit_transform(x_fit_kernel_matrix)

        eigenproblem = self._make_eigenproblem(x_fit_kernel_matrix, context)
        self._fit_transform_in_place(eigenproblem)

        if self.fit_inverse_transform:
            x_transformed = self.transform(context.X_input, covariates=context.covariates_input)
            self._fit_inverse_transform(x_transformed, context.X_input)

        return self

    def transform(self, X, covariates=None):
        """Project new samples into the learned latent space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        covariates : array-like, default=None
            Covariates aligned with ``X``. Provide them in the same raw format
            used at fit time when ``covariate_encoder`` is configured, or in
            the same numeric feature representation otherwise.
        """
        check_is_fitted(self)
        accept_sparse = False if self.fit_inverse_transform else "csr"
        X = validate_data(self, X, accept_sparse=accept_sparse, reset=False)
        covariates = self._transform_covariates(covariates)
        covariates = self._validate_transform_covariates(covariates, X)

        if covariates is None and self.augment in {"pre", "post"}:
            raise ValueError("Covariates must be provided for transform when `augment` is 'pre' or 'post'.")

        if covariates is not None and hasattr(self, "_factor_validator"):
            covariates = self._factor_validator.transform(covariates)

        X_query = self._augment_data(X, covariates)
        x_fit_kernel_matrix = self._get_kernel(X_query, self.x_fit_)
        x_fit_kernel_matrix = self._centerer.transform(x_fit_kernel_matrix)

        w = self.eigenvectors_
        if self.scale_components:
            w = _scale_eigenvectors(self.eigenvalues_, w)

        z = safe_sparse_dot(x_fit_kernel_matrix, w)

        if self.augment == "post":
            z = np.hstack((z, covariates))

        return z

    def inverse_transform(self, z):
        check_is_fitted(self)
        if not self.fit_inverse_transform:
            raise NotFittedError(
                "The fit_inverse_transform parameter was not"
                " set to True when instantiating and hence "
                "the inverse transform is not available."
            )

        k_z = self._get_kernel(z, self.x_transformed_fit_)
        return safe_sparse_dot(k_z, self.dual_coef_)

    def fit_transform(self, X, y=None, covariates=None, **fit_params):
        self.fit(X, y=y, covariates=covariates, **fit_params)
        return self.transform(X, covariates=covariates)


class BaseMMDDomainAdapter(BaseKernelDomainAdapter):
    """Shared base for MMD-based kernel domain adaptation methods.

    ``covariates`` are interpreted as binary domain labels. They must contain
    exactly two values during :meth:`fit`, one for the source domain and one
    for the target domain. ``covariate_encoder`` is intentionally unsupported
    because TCA, JDA, and BDA expect domain labels rather than general side
    information.
    """

    _parameter_constraints: dict = {
        **BaseKernelDomainAdapter._parameter_constraints,
        "lambda_": [Interval(Real, 0, None, closed="left")],
    }

    _eigen_preprocess_steps = ["svd_flip", "sort_eigencomponents"]

    def __init__(
        self,
        n_components=None,
        lambda_=1.0,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        covariate_encoder=None,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        n_jobs=None,
    ):
        if covariate_encoder is not None:
            raise ValueError(
                "`covariate_encoder` is not supported for MMD-based adapters. "
                "TCA, JDA, and BDA expect binary domain covariates."
            )
        self.lambda_ = lambda_
        super().__init__(
            n_components=n_components,
            ignore_y=False,
            augment=None,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            covariate_encoder=None,
            alpha=1.0,
            fit_inverse_transform=False,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            scale_components=scale_components,
            random_state=random_state,
            copy=copy,
            n_jobs=n_jobs,
        )

    def _validate_covariates(self, covariates, X):
        if covariates is None:
            return None, None

        covariates = np.asarray(covariates)
        if covariates.ndim == 0 or covariates.ndim > 2 or (covariates.ndim == 2 and covariates.shape[1] != 1):
            raise ValueError(f"Covariates for {self.__class__.__name__} must be a 1D array of binary domain labels.")
        covariates = covariates.reshape(-1)
        if covariates.shape[0] != _num_samples(X):
            raise ValueError("Covariates and X must have the same number of samples.")
        if not (np.issubdtype(covariates.dtype, np.number) or np.issubdtype(covariates.dtype, np.bool_)):
            raise ValueError(f"Covariates for {self.__class__.__name__} should be numeric or boolean domain labels.")
        if np.unique(covariates).size != 2:
            raise ValueError(
                f"Covariates for {self.__class__.__name__} must contain both source and target domain values."
            )
        return covariates, None

    def _validate_transform_covariates(self, covariates, X):
        if covariates is None:
            return None

        covariates = np.asarray(covariates)
        if covariates.ndim == 0 or covariates.ndim > 2 or (covariates.ndim == 2 and covariates.shape[1] != 1):
            raise ValueError(f"Covariates for {self.__class__.__name__} must be a 1D array of binary domain labels.")

        covariates = covariates.reshape(-1)
        if covariates.shape[0] != _num_samples(X):
            raise ValueError("Covariates and X must have the same number of samples.")
        if not (np.issubdtype(covariates.dtype, np.number) or np.issubdtype(covariates.dtype, np.bool_)):
            raise ValueError(f"Covariates for {self.__class__.__name__} should be numeric or boolean domain labels.")

        return covariates

    def _prepare_fit_context(self, X, y=None, covariates=None, target_covariate=None, **fit_params):
        n_samples = _num_samples(X)
        if covariates is None:
            if y is not None:
                raise ValueError(
                    "Source and target indices are not defined. Please provide covariates and target_covariate."
                )
            _, y_encoded, classes = self._encode_y_for_fit(None, n_samples, X.dtype)
            return KernelFitContext(
                X_input=X,
                X_fit=X,
                y_encoded=y_encoded,
                classes=classes,
            )

        unique_covariates = np.unique(covariates)
        if target_covariate is None:
            target_covariate = unique_covariates[0]
        elif target_covariate not in unique_covariates:
            raise ValueError("`target_covariate` must match one of the observed covariate values.")

        target_idx = np.where(covariates == target_covariate)[0]
        source_idx = np.where(covariates != target_covariate)[0]

        X_fit = np.vstack((X[source_idx], X[target_idx]))
        covariates_fit = np.concatenate((covariates[source_idx], covariates[target_idx]))
        ns = source_idx.shape[0]
        nt = target_idx.shape[0]

        ys = None
        yt = None
        y_fit = None
        classes = None
        unlabeled_value = None
        if y is not None:
            y = np.asarray(y)
            y_labeled_count = _num_samples(y)

            if y_labeled_count == ns:
                ys = y
                target_labeled_pos = np.array([], dtype=int)
            elif n_samples >= y_labeled_count > ns:
                source_labeled_idx = source_idx[source_idx < y_labeled_count]
                if source_labeled_idx.shape[0] != ns:
                    raise ValueError("All source samples must be labeled when y includes target labels.")
                ys = y[source_labeled_idx]
                target_labeled_pos = np.flatnonzero(target_idx < y_labeled_count)
                yt = y[target_idx[target_labeled_pos]] if target_labeled_pos.size > 0 else None
            else:
                raise ValueError("Number of labeled samples does not meet the required conditions.")

            unlabeled_value = self._get_unlabeled_value(y)
            y_fit = np.full(ns + nt, unlabeled_value, dtype=object)
            y_fit[:ns] = ys
            if yt is not None:
                y_fit[ns + target_labeled_pos] = yt

        y_fit, y_encoded, classes = self._encode_y_for_fit(y_fit, ns + nt, X.dtype, unlabeled_value=unlabeled_value)

        return KernelFitContext(
            X_input=X,
            X_fit=X_fit,
            covariates_input=covariates,
            covariates_fit=covariates_fit,
            y_input=y,
            y_fit=y_fit,
            y_encoded=y_encoded,
            classes=classes,
            source_idx=source_idx,
            target_idx=target_idx,
            ns=ns,
            nt=nt,
            ys=ys,
            yt=yt,
            target_covariate=target_covariate,
        )

    def _make_marginal_mmd_matrix(self, context):
        if context.ns is None or context.nt is None or context.target_idx is None:
            return np.zeros((context.X_fit.shape[0], context.X_fit.shape[0]))

        mmd_matrix = mmd_coef(context.ns, context.nt, kind="marginal", mu=0)
        mmd_matrix[np.isnan(mmd_matrix)] = 0
        return mmd_matrix

    def _get_eigenvalue_order(self):
        return "ascending"
