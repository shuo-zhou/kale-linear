import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from kalelinear.transformer import BDA, JDA, MIDA, TCA


@pytest.fixture
def domain_adaptation_data():
    xs = np.array(
        [
            [-2.0, -1.8],
            [-1.8, -2.1],
            [1.9, 1.7],
            [2.1, 2.0],
        ]
    )
    ys = np.array([0, 0, 1, 1])
    xt = np.array(
        [
            [-1.4, -1.2],
            [-1.2, -1.1],
            [1.2, 1.1],
            [1.4, 1.3],
        ]
    )
    yt = np.array([0, 0, 1, 1])
    x = np.vstack((xs, xt))
    y = np.concatenate((ys, yt))
    binary_covariates = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    covariates = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    return xs, ys, xt, yt, x, y, binary_covariates, covariates


@pytest.mark.parametrize("transformer_cls", [TCA, JDA, BDA])
def test_domain_transformers_fit_transform_shapes(transformer_cls, domain_adaptation_data):
    xs, ys, xt, yt, x, y, binary_covariates, _ = domain_adaptation_data
    transformer = transformer_cls(n_components=2)

    x_transformed = transformer.fit_transform(x, y=y, covariates=binary_covariates, target_covariate=1)
    xs_transformed = x_transformed[binary_covariates == 0]
    xt_transformed = x_transformed[binary_covariates == 1]

    assert xs_transformed.shape == (xs.shape[0], 2)
    assert xt_transformed.shape == (xt.shape[0], 2)
    assert np.isfinite(xs_transformed).all()
    assert np.isfinite(xt_transformed).all()


def test_mida_fit_transform_shapes_with_covariates(domain_adaptation_data):
    _, _, _, _, x, y, _, covariates = domain_adaptation_data
    transformer = MIDA(n_components=2)

    x_transformed = transformer.fit_transform(x, y=y, covariates=covariates)

    assert x_transformed.shape == (x.shape[0], 2)
    assert np.isfinite(x_transformed).all()


def test_mida_transform_requires_fit(domain_adaptation_data):
    xs, _, _, _, _, _, _, _ = domain_adaptation_data
    transformer = MIDA(n_components=2)

    with pytest.raises(NotFittedError):
        transformer.transform(xs)


def test_transformers_store_training_projection(domain_adaptation_data):
    xs, _, xt, _, x, y, binary_covariates, covariates = domain_adaptation_data

    tca = TCA(n_components=2).fit(x, y=y, covariates=binary_covariates, target_covariate=1)
    jda = JDA(n_components=2).fit(x, y=y, covariates=binary_covariates, target_covariate=1)
    bda = BDA(n_components=2, mu=0.25).fit(x, y=y, covariates=binary_covariates, target_covariate=1)
    mida = MIDA(n_components=2).fit(x, y=y, covariates=covariates)

    assert tca.U.shape[0] == xs.shape[0] + xt.shape[0]
    assert jda.U.shape[0] == xs.shape[0] + xt.shape[0]
    assert bda.U.shape[0] == xs.shape[0] + xt.shape[0]
    assert mida.U.shape[0] == xs.shape[0] + xt.shape[0]


@pytest.mark.parametrize("transformer_cls", [TCA, JDA, BDA])
def test_mmd_transformers_reject_covariate_encoder(transformer_cls, domain_adaptation_data):
    _, _, _, _, _, _, _, _ = domain_adaptation_data

    with pytest.raises(ValueError, match="covariate_encoder"):
        transformer_cls(n_components=2, covariate_encoder="onehot")


@pytest.mark.parametrize("transformer_cls", [TCA, JDA, BDA])
def test_mmd_transformers_require_both_domains(transformer_cls, domain_adaptation_data):
    _, _, _, _, x, _, _, _ = domain_adaptation_data

    with pytest.raises(ValueError, match="both source and target"):
        transformer_cls(n_components=2).fit(x, covariates=np.zeros(x.shape[0], dtype=int))


@pytest.mark.parametrize("transformer_cls", [TCA, JDA, BDA])
def test_mmd_transformers_validate_target_covariate(transformer_cls, domain_adaptation_data):
    _, _, _, _, x, _, binary_covariates, _ = domain_adaptation_data

    with pytest.raises(ValueError, match="target_covariate"):
        transformer_cls(n_components=2).fit(x, covariates=binary_covariates, target_covariate=2)


def test_jda_does_not_accept_mu():
    with pytest.raises(TypeError, match="mu"):
        JDA(n_components=2, mu=0.5)


@pytest.mark.parametrize("mu", [-0.1, 1.1])
def test_bda_validates_mu(mu, domain_adaptation_data):
    _, _, _, _, x, y, binary_covariates, _ = domain_adaptation_data

    with pytest.raises(ValueError, match="mu"):
        BDA(n_components=2, mu=mu).fit(x, y=y, covariates=binary_covariates, target_covariate=1)
