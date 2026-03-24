import numpy as np
import pytest

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
