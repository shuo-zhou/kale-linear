# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, szhou20@sheffield.ac.uk
# =============================================================================
"""Joint and Balanced Distribution Adaptation."""

from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval

from kalelinear.transformer._base import _num_features, BaseMMDDomainTransformer
from kalelinear.utils import centering_matrix, mmd_coef


class JDA(BaseMMDDomainTransformer):
    """Joint Distribution Adaptation.

    ``covariates`` represent binary domain labels of length ``n_samples``.
    They must contain both source and target domains during :meth:`fit`.
    ``target_covariate`` selects which label is treated as the target domain.
    ``covariate_encoder`` is unsupported because JDA expects domain labels
    rather than general categorical side information.
    """

    def __init__(
        self,
        n_components=None,
        kernel="linear",
        lambda_=1.0,
        covariate_encoder=None,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        n_jobs=None,
        **kwargs,
    ):
        if "mu" in kwargs:
            raise TypeError("JDA.__init__() got an unexpected keyword argument 'mu'")
        super().__init__(
            n_components=n_components,
            lambda_=lambda_,
            kernel=kernel,
            kernel_params=kwargs or None,
            covariate_encoder=covariate_encoder,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            scale_components=scale_components,
            random_state=random_state,
            copy=copy,
            n_jobs=n_jobs,
        )

    def _joint_mmd_mu(self):
        return 0.5

    def _make_eigenproblem(self, x_kernel_matrix, context):
        mmd_matrix = self._make_marginal_mmd_matrix(context)
        if context.ys is not None and context.yt is not None:
            mmd_matrix = mmd_coef(
                context.ns,
                context.nt,
                context.ys,
                context.yt,
                kind="joint",
                mu=self._joint_mmd_mu(),
            )
            mmd_matrix[np.isnan(mmd_matrix)] = 0

        h = centering_matrix(_num_features(x_kernel_matrix), x_kernel_matrix.dtype)
        identity = np.eye(_num_features(x_kernel_matrix), dtype=x_kernel_matrix.dtype)

        obj = x_kernel_matrix @ mmd_matrix @ x_kernel_matrix + self.lambda_ * identity
        st = x_kernel_matrix @ h @ x_kernel_matrix

        return obj, st


class BDA(JDA):
    """Balanced Distribution Adaptation.

    ``covariates`` represent binary domain labels of length ``n_samples``.
    They must contain both source and target domains during :meth:`fit`.
    ``target_covariate`` selects which label is treated as the target domain.
    ``covariate_encoder`` is unsupported because BDA expects domain labels
    rather than general categorical side information.
    """

    _parameter_constraints: dict = {
        **BaseMMDDomainTransformer._parameter_constraints,
        "mu": [Interval(Real, 0, 1, closed="both")],
    }

    def __init__(
        self,
        n_components=None,
        kernel="linear",
        lambda_=1.0,
        mu=0.5,
        covariate_encoder=None,
        eigen_solver="auto",
        tol=0,
        max_iter=None,
        iterated_power="auto",
        remove_zero_eig=False,
        scale_components=False,
        random_state=None,
        copy=True,
        n_jobs=None,
        **kwargs,
    ):
        if mu < 0 or mu > 1:
            raise ValueError("mu should be in the range [0, 1]")
        self.mu = mu
        super().__init__(
            n_components=n_components,
            kernel=kernel,
            lambda_=lambda_,
            covariate_encoder=covariate_encoder,
            eigen_solver=eigen_solver,
            tol=tol,
            max_iter=max_iter,
            iterated_power=iterated_power,
            remove_zero_eig=remove_zero_eig,
            scale_components=scale_components,
            random_state=random_state,
            copy=copy,
            n_jobs=n_jobs,
            **kwargs,
        )

    def _joint_mmd_mu(self):
        return self.mu
