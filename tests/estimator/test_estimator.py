import numpy as np
import pytest
from sklearn.metrics import accuracy_score

import kalelinear.estimator as estimator


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


def test_coir_svm_none_covariates_matches_zero_covariates(office_test_data):
    x, _, tgt_idx, _, x_train, _, y_train = _split_source_target(office_test_data)
    zero_covariates = np.zeros((x_train.shape[0], 1))

    clf_none = estimator.CoIRSVM().fit(x_train, y_train, covariates=None)
    clf_zero = estimator.CoIRSVM().fit(x_train, y_train, covariates=zero_covariates)

    dec_none = clf_none.decision_function(x[tgt_idx])
    dec_zero = clf_zero.decision_function(x[tgt_idx])
    pred_none = clf_none.predict(x[tgt_idx])
    pred_zero = clf_zero.predict(x[tgt_idx])

    assert np.allclose(dec_none, dec_zero)
    assert np.array_equal(pred_none, pred_zero)


def test_coir_ls_none_covariates_matches_zero_covariates(office_test_data):
    x, _, tgt_idx, _, x_train, _, y_train = _split_source_target(office_test_data)
    zero_covariates = np.zeros((x_train.shape[0], 1))

    clf_none = estimator.CoIRLS().fit(x_train, y_train, covariates=None)
    clf_zero = estimator.CoIRLS().fit(x_train, y_train, covariates=zero_covariates)

    dec_none = clf_none.decision_function(x[tgt_idx])
    dec_zero = clf_zero.decision_function(x[tgt_idx])
    pred_none = clf_none.predict(x[tgt_idx])
    pred_zero = clf_zero.predict(x[tgt_idx])

    assert np.allclose(dec_none, dec_zero)
    assert np.array_equal(pred_none, pred_zero)


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


def test_gsda_fit_predicts_target_labels(office_test_data):
    x, y, z, covariate_mat = office_test_data
    target_idx = np.where(z == 0)[0]

    clf = estimator.GSDA(max_iter=25, random_state=0)
    clf.fit(x, y[target_idx], covariate_mat, target_idx=target_idx)

    y_proba = clf.predict_proba(x[target_idx])
    y_pred = clf.predict(x[target_idx])
    params = clf.get_fitted_params()

    assert clf.coef_.shape == (x.shape[1],)
    assert params["coef"].shape == (x.shape[1],)
    assert np.isfinite(clf.coef_).all()
    assert np.isfinite(clf.intercept_)
    assert y_proba.shape == y[target_idx].shape
    assert y_pred.shape == y[target_idx].shape
    assert np.all((0 <= y_proba) & (y_proba <= 1))
    assert set(np.unique(y_pred)).issubset({0.0, 1.0})
    assert len(clf.losses["time"]) == 1
    assert len(clf.losses["ovr"]) > 0


def test_laprls_supports_torch_backend(office_test_data):
    torch = pytest.importorskip("torch")

    x, y, tgt_idx, src_idx, x_train, _, y_train = _split_source_target(office_test_data)
    clf = estimator.LapRLS()

    X_train = torch.tensor(x_train, dtype=torch.float32)
    y_source = torch.tensor(y_train, dtype=torch.int64)
    X_target = torch.tensor(x[tgt_idx], dtype=torch.float32)

    clf.fit(X_train, y_source)
    decision = clf.decision_function(X_target)
    y_pred = clf.predict(X_target)

    assert isinstance(decision, torch.Tensor)
    assert isinstance(y_pred, torch.Tensor)
    assert decision.shape[0] == len(tgt_idx[0])
    assert y_pred.shape[0] == len(tgt_idx[0])
