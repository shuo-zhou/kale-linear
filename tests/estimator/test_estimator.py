import numpy as np
import pytest
from sklearn.metrics import accuracy_score

import pydale.estimator as estimator


def test_svm(office_test_data):
    x, y, z, covariate_mat = office_test_data

    tgt = 0
    tgt_idx = np.where(z == tgt)
    src_idx = np.where(z != tgt)

    x_train = np.concatenate((x[src_idx], x[tgt_idx]))
    c_train = np.concatenate((covariate_mat[src_idx], covariate_mat[tgt_idx]))
    y_train = y[src_idx]

    clf1 = estimator.SIDeRSVM()
    clf2 = estimator.SIDeRSVM(solver="cvxopt")

    clf1.fit(x_train, y_train, c_train)
    clf2.fit(x_train, y_train, c_train)

    clf1.coef_ - clf2.coef_ == 0


@pytest.mark.parametrize("clf", [estimator.SIDeRSVM(), estimator.SIDeRLS()])
def test_sider(clf, office_test_data):
    x, y, z, covariate_mat = office_test_data

    tgt = 0
    tgt_idx = np.where(z == tgt)
    src_idx = np.where(z != tgt)

    x_train = np.concatenate((x[src_idx], x[tgt_idx]))
    c_train = np.concatenate((covariate_mat[src_idx], covariate_mat[tgt_idx]))
    y_train = y[src_idx]

    clf.fit(x_train, y_train, c_train)

    y_pred = clf.predict(x[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert 0 <= acc <= 1


@pytest.mark.parametrize("clf", [estimator.ARSVM(), estimator.ARRLS()])
def test_artl(clf, office_test_data):
    x, y, z, covariate_mat = office_test_data

    tgt = 0
    tgt_idx = np.where(z == tgt)
    src_idx = np.where(z != tgt)

    clf.fit(x[src_idx], y[src_idx], xt=x[tgt_idx])

    y_pred = clf.predict(x[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert 0 <= acc <= 1


@pytest.mark.parametrize("clf", [estimator.LapSVM(), estimator.LapRLS()])
def test_manifold(clf, office_test_data):
    x, y, z, covariate_mat = office_test_data

    tgt = 0
    tgt_idx = np.where(z == tgt)
    src_idx = np.where(z != tgt)

    x_train = np.concatenate((x[src_idx], x[tgt_idx]))
    clf.fit(x_train, y[src_idx])

    y_pred = clf.predict(x[tgt_idx])
    acc = accuracy_score(y[tgt_idx], y_pred)

    assert 0 <= acc <= 1
