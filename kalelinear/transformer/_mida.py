# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================
"""Maximum Independence Domain Adaptation (MIDA) implementation."""

from numbers import Real

from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils._param_validation import Interval

from kalelinear.transformer._base import _centering_kernel, _num_features, BaseKernelDomainAdapter


class MIDA(BaseKernelDomainAdapter):
    """Maximum Independence Domain Adaptation (MIDA).

    MIDA learns a covariate-invariant feature space by maximizing the
    Hilbert-Schmidt independence criterion (HSIC) with respect to the
    provided covariates.

    To prevent label leakage, set target labels to ``-1``.

    ``covariates`` are required during :meth:`fit`. With
    ``covariate_encoder=None``, they must already be numeric and shaped as
    ``(n_samples,)`` or ``(n_samples, n_covariates)``. Use
    ``covariate_encoder="onehot"`` to fit a one-hot encoder on raw
    categorical covariates before adaptation. During :meth:`transform`,
    covariates are only required when ``augment`` is ``"pre"`` or ``"post"``.
    """

    _parameter_constraints: dict = {
        **BaseKernelDomainAdapter._parameter_constraints,
        "mu": [Interval(Real, 0, None, closed="neither")],
        "eta": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        n_components=None,
        mu=1.0,
        eta=1.0,
        ignore_y=False,
        augment=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        covariate_encoder=None,
        alpha=1,
        fit_inverse_transform=False,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        num_jobs=None,
    ):
        self.mu = mu
        self.eta = eta

        super().__init__(
            n_components=n_components,
            ignore_y=ignore_y,
            augment=augment,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
            covariate_encoder=covariate_encoder,
            alpha=alpha,
            fit_inverse_transform=fit_inverse_transform,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            scale_components=scale_components,
            random_state=random_state,
            copy=copy,
            n_jobs=num_jobs,
        )

    def _requires_covariates(self):
        return True

    def _make_eigenproblem(self, x_kernel_matrix, context):
        h = _centering_kernel(_num_features(x_kernel_matrix), x_kernel_matrix.dtype)
        y_kernel_matrix = pairwise_kernels(context.y_encoded, n_jobs=self.n_jobs)
        covariate_kernel = pairwise_kernels(context.covariates_fit, n_jobs=self.n_jobs)

        centerer = KernelCenterer()
        y_kernel_matrix = centerer.fit_transform(y_kernel_matrix)
        covariate_kernel = centerer.fit_transform(covariate_kernel)

        return multi_dot(
            (x_kernel_matrix, self.mu * h + self.eta * y_kernel_matrix - covariate_kernel, x_kernel_matrix)
        )
