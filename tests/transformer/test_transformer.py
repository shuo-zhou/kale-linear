import numpy as np
import pytest

from kalelinear.transformer import JDA, MIDA, TCA


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
    covariates = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    return xs, ys, xt, yt, covariates


@pytest.mark.parametrize("transformer_cls", [TCA, JDA])
def test_domain_transformers_fit_transform_shapes(transformer_cls, domain_adaptation_data):
    xs, ys, xt, yt, _ = domain_adaptation_data
    transformer = transformer_cls(n_components=2)

    xs_transformed, xt_transformed = transformer.fit_transform(xs, ys=ys, Xt=xt, yt=yt)

    assert xs_transformed.shape == (xs.shape[0], 2)
    assert xt_transformed.shape == (xt.shape[0], 2)
    assert np.isfinite(xs_transformed).all()
    assert np.isfinite(xt_transformed).all()


def test_mida_fit_transform_shapes_with_covariates(domain_adaptation_data):
    xs, ys, xt, yt, covariates = domain_adaptation_data
    x = np.vstack((xs, xt))
    y = np.concatenate((ys, yt))
    transformer = MIDA(n_components=2)

    x_transformed = transformer.fit_transform(x, y=y, covariates=covariates)

    assert x_transformed.shape == (x.shape[0], 2)
    assert np.isfinite(x_transformed).all()


def test_mida_transform_requires_fit(domain_adaptation_data):
    xs, _, _, _, _ = domain_adaptation_data
    transformer = MIDA(n_components=2)

    with pytest.raises(Exception):
        transformer.transform(xs)


def test_transformers_store_training_projection(domain_adaptation_data):
    xs, ys, xt, yt, covariates = domain_adaptation_data

    tca = TCA(n_components=2).fit(xs, ys=ys, Xt=xt, yt=yt)
    jda = JDA(n_components=2).fit(xs, ys=ys, Xt=xt, yt=yt)
    mida = MIDA(n_components=2).fit(np.vstack((xs, xt)), y=np.concatenate((ys, yt)), covariates=covariates)

    assert tca.U.shape[0] == xs.shape[0] + xt.shape[0]
    assert jda.U.shape[0] == xs.shape[0] + xt.shape[0]
    assert mida.U.shape[0] == xs.shape[0] + xt.shape[0]
