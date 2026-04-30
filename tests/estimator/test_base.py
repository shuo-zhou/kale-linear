import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from kalelinear.estimator import ARRLS, ARSVM, CoIRLS, CoIRSVM, GSDA, LapRLS, LapSVM
from kalelinear.estimator.base import BaseDomainAdaptationEstimator, BaseKaleEstimator


def test_domain_adaptation_base_inherits_base_kale_estimator():
    assert issubclass(BaseDomainAdaptationEstimator, BaseKaleEstimator)


@pytest.mark.parametrize("estimator_cls", [ARSVM, ARRLS, CoIRSVM, CoIRLS])
def test_domain_adaptation_estimators_inherit_domain_adaptation_base(estimator_cls):
    assert issubclass(estimator_cls, BaseDomainAdaptationEstimator)


@pytest.mark.parametrize("estimator_cls", [GSDA, LapSVM, LapRLS])
def test_core_estimators_inherit_base_kale_estimator(estimator_cls):
    assert issubclass(estimator_cls, BaseKaleEstimator)


def test_base_kale_estimator_decision_function_uses_X_fit_data():
    X_train = np.array([[1.0, 0.0], [0.0, 1.0]])
    estimator = BaseKaleEstimator()
    estimator.coef_ = np.array([2.0, -1.0])
    estimator._lb.fit(
        np.array([0, 1]),
    )
    estimator.X = X_train

    scores = estimator.decision_function(X_train)
    y_pred = estimator.predict(X_train)

    assert np.allclose(scores, np.array([2.0, -1.0]))
    assert y_pred.shape == (2,)


def test_base_kale_estimator_decision_function_requires_fit_data():
    estimator = BaseKaleEstimator()
    estimator.coef_ = np.array([1.0])

    with pytest.raises(NotFittedError):
        estimator.decision_function(np.array([[1.0]]))


def test_quadprog_rejects_invalid_solver():
    with pytest.raises(ValueError, match="Invalid QP solver"):
        BaseKaleEstimator._quadprog(np.eye(2), np.array([1.0, -1.0]), C=1.0, solver="invalid")


def test_semi_binary_dual_support_uses_nonzero_alphas(monkeypatch):
    alpha = np.array([0.0, 1e-10, 0.25, 0.25 - 1e-7])

    def quadprog_stub(cls, P, y, C, solver):
        return alpha

    monkeypatch.setattr(BaseKaleEstimator, "_quadprog", classmethod(quadprog_stub))

    _, support = BaseKaleEstimator._semi_binary_dual(
        np.eye(4),
        np.array([1.0, -1.0, 1.0, -1.0]),
        np.eye(4),
        C=1.0,
    )

    assert np.array_equal(support, np.array([2, 3]))
