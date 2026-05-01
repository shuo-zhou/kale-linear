import numpy as np
import pytest
from sklearn.utils.validation import check_random_state

from kalelinear.utils import base_init, centered_kernel_matrix, centering_matrix, hsic_grad_term, lap_norm, mmd_coef


@pytest.fixture
def sample_data():
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )


def test_base_init_returns_expected_shapes(sample_data):
    x_kernel_matrix, unit_matrix, centering_matrix, n = base_init(sample_data)

    assert x_kernel_matrix.shape == (4, 4)
    assert unit_matrix.shape == (4, 4)
    assert centering_matrix.shape == (4, 4)
    assert n == 4
    assert np.allclose(unit_matrix, np.eye(4))


@pytest.mark.parametrize("mode", ["distance", "connectivity"])
def test_lap_norm_returns_square_matrix(sample_data, mode):
    lap = lap_norm(sample_data, n_neighbour=2, mode=mode, normalise=False)

    assert lap.shape == (4, 4)
    assert np.allclose(lap, lap.T)


def test_mmd_coef_returns_square_matrix():
    mmd = mmd_coef(3, 2)
    assert mmd.shape == (5, 5)


def test_mmd_coef_raises_for_mismatched_labels():
    ys = np.array([0, 1, 1])
    yt = np.array([0, 2])

    with pytest.raises(ValueError, match="same labels"):
        mmd_coef(3, 2, ys=ys, yt=yt, kind="joint")


def test_centered_kernel_matrix_matches_manual_centering(sample_data):
    centered_kernel = centered_kernel_matrix(sample_data)
    manual_centering = centering_matrix(sample_data.shape[0])
    manual_kernel = sample_data @ sample_data.T

    assert np.allclose(centered_kernel, manual_centering @ manual_kernel @ manual_centering)


def test_hsic_grad_term_matches_manual_linear_covariate_form(sample_data):
    w = np.array([0.25, -0.5])
    covariates = np.array([[0.0], [0.0], [1.0], [1.0]])
    centered_covariate_kernel = centered_kernel_matrix(covariates)

    grad_term = hsic_grad_term(w, sample_data, covariates)

    assert np.allclose(grad_term, sample_data.T @ centered_covariate_kernel @ sample_data @ w)


def make_domain_shifted_dataset(
    num_domains=3,
    num_samples_per_class=100,
    num_features=2,
    class_sep=1.0,
    centroid_shift_scale=5.0,
    random_state=None,
):
    random_state = check_random_state(random_state)

    # Shared linear separator
    w = random_state.randn(num_features)
    w = w / np.linalg.norm(w)

    x_all = []
    y_all = []
    domain_all = []

    for i_domain in range(num_domains):
        domain_shift = random_state.randn(num_features) * centroid_shift_scale

        for label in [0, 1]:
            class_mean = (label - 0.5) * class_sep * w + domain_shift
            cov = np.eye(num_features)
            x_class = random_state.multivariate_normal(class_mean, cov, num_samples_per_class)
            y_class = np.full(num_samples_per_class, label)
            domain_class = np.full(num_samples_per_class, i_domain)

            x_all.append(x_class)
            y_all.append(y_class)
            domain_all.append(domain_class)

    x = np.vstack(x_all)
    y = np.concatenate(y_all)
    domains = np.concatenate(domain_all)

    idx = random_state.permutation(len(x))
    x = x[idx]
    y = y[idx]
    domains = domains[idx]

    return x, y, domains
