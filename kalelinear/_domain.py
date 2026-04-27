"""Private helpers for source/target domain bookkeeping."""

from dataclasses import dataclass

import numpy as np


@dataclass
class DomainSplit:
    """Source/target partition derived from binary domain covariates."""

    source_idx: np.ndarray
    target_idx: np.ndarray
    target_covariate: object


def check_binary_domain_covariates(
    covariates,
    n_samples,
    *,
    require_numeric=False,
    error_prefix="`covariates`",
    both_domains_message=None,
):
    """Validate and normalize binary source/target covariates."""
    covariates = np.asarray(covariates)
    if covariates.ndim == 2 and covariates.shape[1] == 1:
        covariates = covariates.reshape(-1)
    if covariates.ndim != 1:
        raise ValueError(f"{error_prefix} must be a 1D array of binary source/target domain labels.")
    if covariates.shape[0] != n_samples:
        raise ValueError(f"{error_prefix} and X must have the same number of samples.")
    if require_numeric and not (
        np.issubdtype(covariates.dtype, np.number) or np.issubdtype(covariates.dtype, np.bool_)
    ):
        raise ValueError(f"{error_prefix} should be numeric or boolean domain labels.")

    unique_covariates = np.unique(covariates)
    if unique_covariates.size != 2:
        if both_domains_message is not None:
            raise ValueError(both_domains_message)
        raise ValueError(f"{error_prefix} must contain exactly two domain values.")

    return covariates, unique_covariates


def split_domain_indices(covariates, target_covariate=None, *, unique_covariates=None):
    """Return source and target indices from normalized binary covariates."""
    if unique_covariates is None:
        unique_covariates = np.unique(covariates)

    if target_covariate is None:
        target_covariate = unique_covariates[0]
    elif target_covariate not in unique_covariates:
        raise ValueError("`target_covariate` must match one of the observed covariate values.")

    target_idx = np.flatnonzero(covariates == target_covariate)
    source_idx = np.flatnonzero(covariates != target_covariate)
    return DomainSplit(source_idx=source_idx, target_idx=target_idx, target_covariate=target_covariate)


def split_source_target(X, covariates, target_covariate=None, *, require_numeric=False, error_prefix="`covariates`"):
    """Validate covariates and return source/target indices for X."""
    covariates, unique_covariates = check_binary_domain_covariates(
        covariates,
        X.shape[0],
        require_numeric=require_numeric,
        error_prefix=error_prefix,
    )
    return split_domain_indices(covariates, target_covariate, unique_covariates=unique_covariates)
