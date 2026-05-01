# =============================================================================
# @author: Shuo Zhou, The University of Sheffield, shuo.zhou@sheffield.ac.uk
# =============================================================================
"""Transfer Component Analysis (TCA)."""

import numpy as np
from numpy.linalg import multi_dot

from kalelinear.transformer._base import _num_features, BaseMMDDomainTransformer
from kalelinear.utils import centering_matrix, lap_norm


class TCA(BaseMMDDomainTransformer):
    """Transfer Component Analysis.

    ``covariates`` represent binary domain labels of length ``n_samples``.
    They must contain both source and target domains during :meth:`fit`.
    ``target_covariate`` selects which label is treated as the target domain.
    ``covariate_encoder`` is unsupported because TCA expects domain labels
    rather than general categorical side information.
    """

    def __init__(
        self,
        n_components=None,
        kernel="linear",
        lambda_=1.0,
        mu=1.0,
        gamma_=0.5,
        k=3,
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
        self.mu = mu
        self.gamma_ = gamma_
        self.k = k
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

    def _make_eigenproblem(self, x_kernel_matrix, context):
        h = centering_matrix(_num_features(x_kernel_matrix), x_kernel_matrix.dtype)
        identity = np.eye(_num_features(x_kernel_matrix), dtype=x_kernel_matrix.dtype)
        mmd_matrix = self._make_marginal_mmd_matrix(context)

        obj = self.lambda_ * identity
        st = x_kernel_matrix @ h @ x_kernel_matrix

        if context.ys is None:
            obj += x_kernel_matrix @ mmd_matrix @ x_kernel_matrix
            return obj, st

        y_kernel_matrix = self.gamma_ * np.dot(context.y_encoded, context.y_encoded.T) + (1 - self.gamma_) * identity
        lap_mat = lap_norm(context.X_fit, n_neighbour=self.k, mode="connectivity")
        obj += multi_dot([x_kernel_matrix, (mmd_matrix + self.mu * lap_mat), x_kernel_matrix])
        st += multi_dot([x_kernel_matrix, h, y_kernel_matrix, h, x_kernel_matrix])

        return obj, st
