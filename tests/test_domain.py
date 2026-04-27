import numpy as np
import pytest

from kalelinear._domain import check_binary_domain_covariates, split_source_target


def test_check_binary_domain_covariates_accepts_column_vector():
    covariates, unique_covariates = check_binary_domain_covariates(np.array([[0], [1], [0]]), 3)

    assert np.array_equal(covariates, np.array([0, 1, 0]))
    assert np.array_equal(unique_covariates, np.array([0, 1]))


def test_check_binary_domain_covariates_rejects_missing_domain():
    with pytest.raises(ValueError, match="exactly two domain values"):
        check_binary_domain_covariates(np.array([0, 0, 0]), 3)


def test_split_source_target_defaults_to_first_sorted_covariate_as_target():
    X = np.zeros((4, 2))
    split = split_source_target(X, np.array([2, 1, 2, 1]))

    assert split.target_covariate == 1
    assert np.array_equal(split.source_idx, np.array([0, 2]))
    assert np.array_equal(split.target_idx, np.array([1, 3]))


def test_split_source_target_rejects_unknown_target_covariate():
    X = np.zeros((4, 2))

    with pytest.raises(ValueError, match="target_covariate"):
        split_source_target(X, np.array([0, 0, 1, 1]), target_covariate=2)
