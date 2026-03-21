import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from kalinear.estimator.base import BaseFramework


@pytest.mark.parametrize("fit_attr", ["X", "x"])
def test_base_framework_decision_function_uses_fit_data_aliases(fit_attr):
    x_train = np.array([[1.0, 0.0], [0.0, 1.0]])
    estimator = BaseFramework()
    estimator.coef_ = np.array([2.0, -1.0])
    estimator._lb.fit(np.array([0, 1]))
    setattr(estimator, fit_attr, x_train)

    scores = estimator.decision_function(x_train)
    y_pred = estimator.predict(x_train)

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
