import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from kalelinear.estimator.base import BaseFramework


def test_base_framework_decision_function_uses_X_fit_data():
    X_train = np.array([[1.0, 0.0], [0.0, 1.0]])
    estimator = BaseFramework()
    estimator.coef_ = np.array([2.0, -1.0])
    estimator._lb.fit(
        np.array([0, 1]),
    )
    estimator.X = X_train

    scores = estimator.decision_function(X_train)
    y_pred = estimator.predict(X_train)

    assert np.allclose(scores, np.array([2.0, -1.0]))
    assert y_pred.shape == (2,)


def test_base_framework_decision_function_requires_fit_data():
    estimator = BaseFramework()
    estimator.coef_ = np.array([1.0])

    with pytest.raises(NotFittedError):
        estimator.decision_function(np.array([[1.0]]))


def test_quadprog_rejects_invalid_solver():
    with pytest.raises(ValueError, match="Invalid QP solver"):
        BaseFramework._quadprog(np.eye(2), np.array([1.0, -1.0]), C=1.0, solver="invalid")


def test_semi_binary_dual_support_uses_nonzero_alphas(monkeypatch):
    alpha = np.array([0.0, 1e-10, 0.25, 0.25 - 1e-7])

    def quadprog_stub(cls, P, y, C, solver):
        return alpha

    monkeypatch.setattr(BaseFramework, "_quadprog", classmethod(quadprog_stub))

    _, support = BaseFramework._semi_binary_dual(
        np.eye(4),
        np.array([1.0, -1.0, 1.0, -1.0]),
        np.eye(4),
        C=1.0,
    )

    assert np.array_equal(support, np.array([2, 3]))


def test_base_framework_decision_function_supports_torch_backend():
    torch = pytest.importorskip("torch")

    X_train = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    estimator = BaseFramework()
    estimator.coef_ = np.array([2.0, -1.0])
    estimator._lb.fit(
        np.array([0, 1]),
    )
    estimator.X = X_train

    scores = estimator.decision_function(X_train)
    y_pred = estimator.predict(X_train)

    assert isinstance(scores, torch.Tensor)
    assert isinstance(y_pred, torch.Tensor)
    assert scores.shape == (2,)
    assert y_pred.shape == (2,)
