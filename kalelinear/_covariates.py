"""Private helpers for validating and encoding side-information covariates."""

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder


def to_dense_covariates(covariates):
    """Return dense covariates for downstream NumPy matrix operations."""
    if hasattr(covariates, "toarray"):
        return covariates.toarray()
    return covariates


def check_tabular_covariates(covariates, n_samples=None, *, allow_none=True, error_prefix="Covariates"):
    """Validate raw 1D/2D tabular covariates, including strings."""
    if covariates is None:
        if allow_none:
            return None
        raise ValueError(f"{error_prefix} must be provided.")

    covariates = np.asarray(covariates)
    if covariates.ndim == 0 or covariates.ndim > 2:
        raise ValueError(f"{error_prefix} must be a 1D or 2D array-like object.")
    if covariates.ndim == 1:
        covariates = np.expand_dims(covariates, 1)
    if n_samples is not None and covariates.shape[0] != n_samples:
        raise ValueError(f"{error_prefix} and X must have the same number of samples.")
    return covariates


def check_numeric_covariates(
    covariates,
    n_samples=None,
    *,
    allow_none=True,
    error_prefix="Covariates",
    numeric_error_message=None,
):
    """Validate model-ready numeric or boolean covariates."""
    covariates = check_tabular_covariates(
        covariates,
        n_samples,
        allow_none=allow_none,
        error_prefix=error_prefix,
    )
    if covariates is None:
        return None

    covariate_dtype = covariates.dtype
    if not (np.issubdtype(covariate_dtype, np.number) or np.issubdtype(covariate_dtype, np.bool_)):
        if numeric_error_message is not None:
            raise ValueError(numeric_error_message)
        raise ValueError(f"{error_prefix} must be numeric or boolean.")
    return covariates


def make_covariate_encoder(covariate_encoder):
    """Create a covariate encoder from a supported shortcut or estimator."""
    if covariate_encoder == "onehot":
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return clone(covariate_encoder)


def fit_covariate_encoder(covariates, covariate_encoder, n_samples=None, *, error_prefix="Covariates"):
    """Fit an optional covariate encoder and return encoded dense covariates."""
    if covariates is None:
        return None, None
    if covariate_encoder is None:
        return covariates, None

    covariates_to_encode = check_tabular_covariates(covariates, n_samples, error_prefix=error_prefix)
    encoder = make_covariate_encoder(covariate_encoder)
    encoder.fit(covariates_to_encode)
    return to_dense_covariates(encoder.transform(covariates_to_encode)), encoder


def transform_covariates(covariates, encoder, n_samples=None, *, error_prefix="Covariates"):
    """Transform raw covariates with a fitted encoder when one exists."""
    if covariates is None:
        return None
    if encoder is None:
        return covariates

    covariates_to_encode = check_tabular_covariates(covariates, n_samples, error_prefix=error_prefix)
    return to_dense_covariates(encoder.transform(covariates_to_encode))
