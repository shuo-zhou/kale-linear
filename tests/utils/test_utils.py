import numpy as np
import pytest
from sklearn.utils.validation import check_random_state

from kalelinear.utils import base_init, lap_norm, mmd_coef
from kalelinear.utils.multiclass import score2pred


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
    ker_x, unit_mat, ctr_mat, n = base_init(sample_data)

    assert ker_x.shape == (4, 4)
    assert unit_mat.shape == (4, 4)
    assert ctr_mat.shape == (4, 4)
    assert n == 4
    assert np.allclose(unit_mat, np.eye(4))


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


def test_score2pred_selects_top_class_per_row():
    scores = np.array(
        [
            [0.1, 0.8, 0.2],
            [0.4, 0.3, 0.9],
        ]
    )

    pred = score2pred(scores)

    expected = np.array(
        [
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    assert np.array_equal(pred, expected)


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
