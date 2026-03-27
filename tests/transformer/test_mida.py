import numpy as np
import pytest
from numpy import testing
from sklearn.preprocessing import OneHotEncoder

from kalelinear.transformer import MIDA
from tests.utils.test_utils import make_domain_shifted_dataset

N_COMP_CONSTANT = 8


@pytest.fixture(scope="module")
def sample_data():
    # Test an extreme case of domain shift
    # yet the data's manifold is linearly separable
    x, y, domains = make_domain_shifted_dataset(
        num_domains=10,
        num_samples_per_class=2,
        num_features=20,
        centroid_shift_scale=32768,
        random_state=0,
    )

    factors = OneHotEncoder(handle_unknown="ignore").fit_transform(domains.reshape(-1, 1)).toarray()

    return x, y, domains, factors


@pytest.mark.parametrize("num_components", [2, None])
def test_mida_shape_consistency(sample_data, num_components):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=num_components)
    mida.set_params(**mida.get_params())
    mida.fit(x, group_labels=factors)

    # Transform the whole data
    z = mida.transform(x, group_labels=factors)

    # If num_components is not None, check the shape of the transformed data
    if num_components is not None:
        testing.assert_equal(z.shape, (len(x), num_components))

    # Transform the source and target domain data separately
    (source_mask,) = np.where(domains != 0)
    z_src = mida.transform(x[source_mask], group_labels=factors[source_mask])
    z_tgt = mida.transform(x[~source_mask], group_labels=factors[~source_mask])
    # Check if transformations are consistent with separate domains
    testing.assert_allclose(z_src, z[source_mask])
    testing.assert_allclose(z_tgt, z[~source_mask])

    orig_coef_dim = mida.orig_coef_.shape[1]
    feature_dim = x.shape[1]
    assert mida.orig_coef_ is not None, "MIDA must have `orig_coef_` after fitting when kernel='linear'"
    assert orig_coef_dim == feature_dim, f"orig_coef_ shape mismatch: {orig_coef_dim} != {feature_dim}"


def test_mida_inverse_transform(sample_data):
    x, y, domains, factors = sample_data

    mida = MIDA(fit_inverse_transform=True)
    mida.fit(x, group_labels=factors)

    # Transform the whole data
    z = mida.transform(x, group_labels=factors)
    # Inverse transform the data
    x_rec = mida.inverse_transform(z)

    # We don't check whether the inverse transform is exactly equal to the original data
    # in terms of value since it is expected to be different due to the domain adaptation effect.
    # We only check the shape and dimensionality.
    assert len(x_rec) == len(x), f"Inverse transform failed: {len(x_rec)} != {len(x)}"
    assert x_rec.ndim == x.ndim, f"Inverse transform failed: {x_rec.ndim} != {x.ndim}"
    testing.assert_equal(x_rec.shape, x.shape)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "cosine"])
def test_mida_support_kernel(sample_data, kernel):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=N_COMP_CONSTANT, kernel=kernel)
    mida.fit(x, group_labels=factors)

    is_linear = kernel == "linear"
    try:
        has_orig_coef = hasattr(mida, "orig_coef_")
        assert has_orig_coef, "MIDA must have `orig_coef_` after fitting when kernel='linear'"
    except NotImplementedError:
        assert not is_linear, "MIDA must not have `orig_coef_` after fitting when kernel!='linear'"

    # Transform the whole data
    z = mida.transform(x, group_labels=factors)

    # expect to allow multiple kernels supported
    assert mida._n_features_out == N_COMP_CONSTANT, f"Expected {N_COMP_CONSTANT} components, got {mida._n_features_out}"
    assert z.shape[1] == mida._n_features_out, f"Expected {z.shape[1]} features, got {mida._n_features_out}"


@pytest.mark.parametrize("augment", ["pre", "post", None])
def test_mida_augment(sample_data, augment):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=N_COMP_CONSTANT, kernel="linear", augment=augment, fit_inverse_transform=True)
    mida.fit(x, group_labels=factors)

    # expect validator for domain factors if augment=True
    has_validator = hasattr(mida, "_factor_validator")
    if augment:
        assert has_validator, "MIDA must have `_factor_validator` after fitting when augment=True"
        return

    assert not has_validator, "MIDA must not have `_factor_validator` after fitting when augment=False"

    if augment == "post":
        z = mida.transform(x, group_labels=factors)
        assert (
            z.shape[1] == N_COMP_CONSTANT + factors.shape[-1]
        ), f"Expected {x.shape[1]} features, got {N_COMP_CONSTANT + factors.shape[-1]}"


@pytest.mark.parametrize("ignore_y", [True, False])
def test_mida_ignore_y(sample_data, ignore_y):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=N_COMP_CONSTANT, kernel="linear", ignore_y=ignore_y)
    mida.fit(x, y, group_labels=factors)

    # expect classes_ to be set if ignore_y=False
    has_classes = hasattr(mida, "classes_")
    if ignore_y:
        assert not has_classes, "MIDA must not have `classes_` after fitting when ignore_y=True"
        return

    assert has_classes, "MIDA must have `classes_` after fitting when ignore_y=False"


@pytest.mark.parametrize("eigen_solver", ["auto", "dense", "arpack", "randomized"])
def test_mida_eigen_solver(sample_data, eigen_solver):
    x, y, domains, factors = sample_data

    mida = MIDA(
        num_components=N_COMP_CONSTANT,
        kernel="rbf",
        eigen_solver=eigen_solver,
        max_iter=200 if eigen_solver == "arpack" else None,
    )
    mida.fit(x, group_labels=factors)

    # Transform the whole data
    z = mida.transform(x, group_labels=factors)

    # expect the solver to have a consistent number of components
    assert mida._n_features_out == N_COMP_CONSTANT, f"Expected {N_COMP_CONSTANT} components, got {mida._n_features_out}"
    assert z.shape[1] == mida._n_features_out, f"Expected {x.shape[1]} features, got {mida._n_features_out}"


@pytest.mark.parametrize("scale_components", [True, False])
def test_mida_scale_components(sample_data, scale_components):
    x, y, domains, factors = sample_data

    mida = MIDA(num_components=N_COMP_CONSTANT, kernel="linear", scale_components=scale_components)
    mida.fit(x, group_labels=factors)

    # Transform the whole data
    z = mida.transform(x, group_labels=factors)

    # Expect the scale_components to have a consistent number of components
    # the behavior expected is the zero eigenvalues component is masked, not indexed
    assert mida._n_features_out == N_COMP_CONSTANT, f"Expected {N_COMP_CONSTANT} components, got {mida._n_features_out}"
    assert z.shape[1] == mida._n_features_out, f"Expected {x.shape[1]} features, got {mida._n_features_out}"
