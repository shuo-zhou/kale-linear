import numpy as np
import pytest

from kalelinear._covariates import check_numeric_covariates, check_tabular_covariates, fit_covariate_encoder


def test_check_tabular_covariates_accepts_strings():
    covariates = check_tabular_covariates(np.array(["source", "target"]), 2)

    assert covariates.shape == (2, 1)
    assert covariates.dtype.kind in {"O", "U", "S"}


def test_check_numeric_covariates_rejects_strings_without_encoder():
    with pytest.raises(ValueError, match="numeric or boolean"):
        check_numeric_covariates(np.array(["source", "target"]), 2)


def test_fit_covariate_encoder_onehot_encodes_strings():
    covariates, encoder = fit_covariate_encoder(np.array(["source", "target", "source"]), "onehot", 3)

    assert encoder is not None
    assert covariates.shape == (3, 2)
    assert np.issubdtype(covariates.dtype, np.number)
