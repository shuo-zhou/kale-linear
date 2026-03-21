import numpy as np
import pytest
from sklearn.metrics import accuracy_score

import kalinear.estimator as estimator


@pytest.fixture
def office_test_data():
    x = np.array(
        [
            [-2.2, -1.9],
            [-1.9, -2.1],
            [-1.7, -1.8],
            [1.8, 2.1],
            [2.0, 1.9],
            [2.2, 2.0],
            [-1.4, -1.2],
            [-1.1, -1.3],
            [1.3, 1.1],
            [1.5, 1.2],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
    z = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    covariate_mat = np.eye(2)[z]
    return x, y, z, covariate_mat


def _split_source_target(office_test_data, target_domain=0):
    x, y, z, covariate_mat = office_test_data
    tgt_idx = np.where(z == target_domain)
    src_idx = np.where(z != target_domain)
    x_train = np.concatenate((x[src_idx], x[tgt_idx]))
    c_train = np.concatenate((covariate_mat[src_idx], covariate_mat[tgt_idx]))
    y_train = y[src_idx]
    return x, y, tgt_idx, src_idx, x_train, c_train, y_train


def test_coir_svm_solvers_fit_consistently(office_test_data):
    x, y, tgt_idx, src_idx, x_train, c_train, y_train = _split_source_target(office_test_data)

    clf1 = estimator.CoIRSVM()
    clf2 = estimator.CoIRSVM(solver="cvxopt")

    clf1.fit(x_train, y_train, c_train)
    clf2.fit(x_train, y_train, c_train)

    for clf in (clf1, clf2):
        y_pred = clf.predict(x[tgt_idx])
        assert clf.coef_.shape[0] == x_train.shape[0]
        assert np.isfinite(clf.coef_).all()
        assert set(np.unique(y_pred)).issubset(set(np.unique(y)))


@pytest.mark.parametrize("estimator_cls", [estimator.CoIRSVM, estimator.CoIRLS])
def test_coir_estimators_predict_labels(estimator_cls, office_test_data):
    x, y, tgt_idx, src_idx, x_train, c_train, y_train = _split_source_target(office_test_data)

    clf = estimator_cls()
    clf.fit(x_train, y_train, c_train)

    decision = clf.decision_function(x[tgt_idx])
    y_pred = clf.predict(x[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert decision.shape[0] == len(tgt_idx[0])
    assert y_pred.shape == y[tgt_idx].shape
    assert 0 <= acc <= 1


@pytest.mark.parametrize("estimator_cls", [estimator.ARSVM, estimator.ARRLS])
def test_artl_estimators_predict_labels(estimator_cls, office_test_data):
    x, y, tgt_idx, src_idx, _, _, _ = _split_source_target(office_test_data)

    clf = estimator_cls()
    clf.fit(x[src_idx], y[src_idx], Xt=x[tgt_idx])

    decision = clf.decision_function(x[tgt_idx])
    y_pred = clf.predict(x[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert decision.shape[0] == len(tgt_idx[0])
    assert y_pred.shape == y[tgt_idx].shape
    assert 0 <= acc <= 1


@pytest.mark.parametrize("estimator_cls", [estimator.LapSVM, estimator.LapRLS])
def test_manifold_estimators_predict_labels(estimator_cls, office_test_data):
    x, y, tgt_idx, src_idx, x_train, _, y_train = _split_source_target(office_test_data)

    clf = estimator_cls()
    clf.fit(x_train, y_train)

    decision = clf.decision_function(x[tgt_idx])
    y_pred = clf.predict(x[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert decision.shape[0] == len(tgt_idx[0])
    assert y_pred.shape == y[tgt_idx].shape
    assert 0 <= acc <= 1
