# =============================================================================
# @author: Shuo Zhou, The University of Sheffield
# =============================================================================
"""Maximum Independence Domain Adaptation (MIDA) implementation."""

from numbers import Real

from numpy.linalg import multi_dot
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils._param_validation import Interval

# from kalelinear.utils import base_init, infer_backend, to_backend, to_numpy
from kalelinear.transformer._base import _centering_kernel, _num_features, BaseKernelDomainAdapter

# class MIDA(BaseEstimator, TransformerMixin):
#     """Maximum Independence Domain Adaptation (MIDA)

#     A kernel-based domain adaptation method that removes the effect of
#     factors/covariates from the data by learning a feature space derived from maximizing
#     Hilbert-Schmidt independence criterion (HSIC).

#     Parameters
#     ----------
#     n_components : int
#         Number of components after MIDA transformation (n_components <= d)
#     kernel : str, default='linear'
#         'rbf' | 'linear' | 'poly' (default is 'linear')
#     penalty : str, default=None
#         None | 'l2' (default is None) - regularization penalty
#     lambda_ : float, default=1.0
#         Regularization parameter (if penalty=='l2')
#     mu : float, default=1.0
#         L2 kernel regularization coefficient / total captured variance parameter
#     eta : float, default=1.0
#         Class-dependency / label dependence regularization parameter
#     aug : bool, default=True
#         Whether to augment input data with covariates/group_labels.
#         For forward compatibility, also accepts 'pre' or 'post' strings.
#     gamma : float, optional
#         Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Default is None.
#     degree : int, default=3
#         Degree of the polynomial kernel. Ignored by other kernels.
#     coef0 : float, default=1
#         Independent term in the polynomial and sigmoid kernels.
#     ignore_y : bool, default=False
#         Whether to ignore the target variable `y` during fitting.
#     **kwargs
#         Additional kernel parameters passed to pairwise_kernels

#     References
#     ----------
#     Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace
#     using domain features and independence maximization. IEEE transactions on
#     cybernetics, 48(1), pp.288-299.

#     Examples
#     --------
#     >>> import numpy as np
#     >>> from kalelinear.transformer import MIDA
#     >>> # Generate random synthetic data
#     >>> x_source = np.random.normal(loc=5, scale=1, size=(20, 40))
#     >>> x_target = np.random.normal(loc=-5, scale=1, size=(20, 40))
#     >>> y = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
#     >>> # Concatenate source and target data
#     >>> x = np.vstack((x_source, x_target))
#     >>> # Create covariates (e.g., one-hot encoded domain labels)
#     >>> covariates = np.concatenate((np.zeros((20, 1)), np.ones((20, 1))), axis=0)
#     >>> mida = MIDA(n_components=10)
#     >>> x_projected = mida.fit_transform(x, y, covariates=covariates)
#     >>> x_projected.shape
#     (40, 10)
#     """

#     def __init__(
#         self,
#         n_components,
#         penalty=None,
#         kernel="linear",
#         lambda_=1.0,
#         mu=1.0,
#         eta=1.0,
#         aug=True,
#         gamma=None,
#         degree=3,
#         coef0=1,
#         ignore_y=False,
#         **kwargs,
#     ):
#         self.n_components = n_components
#         self.kernel = kernel
#         self.lambda_ = lambda_
#         self.penalty = penalty
#         self.mu = mu
#         self.eta = eta
#         # Support both bool and string for aug parameter
#         self.aug = aug if isinstance(aug, bool) else (aug in ("pre", "post"))
#         self.gamma = gamma
#         self.degree = degree
#         self.coef0 = coef0
#         self.ignore_y = ignore_y
#         self._lb = LabelBinarizer(pos_label=1, neg_label=0)
#         self.kwargs = kwargs

#     def fit(self, X, y=None, covariates=None, group_labels=None):
#         """Fit the MIDA model.

#         Parameters
#         ----------
#         X : array-like
#             Input data, shape (n_samples, n_features)
#         y : array-like, optional
#             Labels, shape (n_samples,). For semi-supervised MIDA, set -1 for unlabeled samples.
#             Default is None for unsupervised learning.
#         covariates : array-like, optional
#             Domain covariates, shape (n_samples, n_covariates).
#             Alias for group_labels for backward compatibility.
#         group_labels : array-like, optional
#             Categorical variables representing domain or grouping factors,
#             shape (n_samples, n_factors). Default is None.

#         Returns
#         -------
#         self : MIDA
#             Returns the fitted estimator.

#         Notes
#         -----
#         Unsupervised MIDA is performed if y is not given.
#         Semi-supervised MIDA is performed if y is given.
#         The covariates (or group_labels) parameter captures domain or factor information.
#         """
#         # Support both covariates and group_labels parameter names
#         factors = group_labels if group_labels is not None else covariates

#         self.backend_ = infer_backend(X, y, factors)
#         X = to_numpy(X)
#         y = to_numpy(y)
#         factors = to_numpy(factors)

#         # Augment X with factors if requested
#         X_aug = X
#         if self.aug and factors is not np.ndarray and type(factors) is np.ndarray:
#             X_aug = np.concatenate((X, factors), axis=1)
#         elif self.aug and factors is not None:
#             X_aug = np.concatenate((X, factors), axis=1)

#         # Initialize kernel and matrices
#         ker_x, unit_mat, ctr_mat, n = base_init(
#             X_aug, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0, **self.kwargs
#         )

#         # Compute kernel for factors
#         if factors is not None and type(factors) is np.ndarray:
#             ker_c = np.dot(factors, factors.T)
#             # Normalize kernel
#             ker_c = ker_c / np.max(np.abs(ker_c)) if np.max(np.abs(ker_c)) > 0 else ker_c
#         else:
#             ker_c = np.zeros((n, n))

#         # Build objective and constraint matrices
#         if y is not None and not self.ignore_y:
#             # Semi-supervised MIDA
#             y_mat = self._lb.fit_transform(y)
#             ker_y = np.dot(y_mat, y_mat.T)

#             # Objective: maximize independence from factors
#             obj = multi_dot([ker_x, ctr_mat, ker_c, ctr_mat, ker_x.T])

#             # Constraint: include label and factor information
#             st = multi_dot([ker_x, ctr_mat, (self.mu * unit_mat + self.eta * ker_y), ctr_mat, ker_x.T])
#         else:
#             # Unsupervised MIDA
#             obj = multi_dot([ker_x, ctr_mat, ker_c, ctr_mat, ker_x.T])
#             st = multi_dot([ker_x, ctr_mat, ker_x.T])

#         # Apply L2 penalty if requested
#         if self.penalty == "l2":
#             obj = obj - self.lambda_ * unit_mat

#         # Generalized eigendecomposition
#         eig_values, eig_vectors = eig(obj, st)
#         idx_sorted = np.argsort(eig_values)

#         self.U = np.asarray(eig_vectors[:, idx_sorted], dtype=np.float64)
#         self.eig_values_ = eig_values[idx_sorted]
#         self.X_fit_ = X_aug

#         return self

#     def transform(self, X, covariates=None, group_labels=None):
#         """Transform data using the fitted MIDA model.

#         Parameters
#         ----------
#         X : array-like
#             Input data to transform, shape (n_samples, n_features)
#         covariates : array-like, optional
#             Domain covariates, shape (n_samples, n_covariates).
#             Alias for group_labels for backward compatibility.
#         group_labels : array-like, optional
#             Categorical variables representing domain or grouping factors,
#             shape (n_samples, n_factors). Required if training data had factors.

#         Returns
#         -------
#         X_transformed : array-like
#             Transformed data, shape (n_samples, n_components)
#         """
#         check_is_fitted(self, "U")

#         # Support both parameter names
#         factors = group_labels if group_labels is not None else covariates

#         backend = infer_backend(X, factors)
#         X_input = X
#         X = to_numpy(X)
#         factors = to_numpy(factors)

#         # Augment X with factors if needed
#         X_aug = X
#         if self.aug and factors is not None and type(factors) is np.ndarray:
#             X_aug = np.concatenate((X, factors), axis=1)

#         # Compute kernel between test data and training data
#         kernel_params = {}
#         if self.gamma is not None:
#             kernel_params["gamma"] = self.gamma
#         if self.degree is not None:
#             kernel_params["degree"] = self.degree
#         if self.coef0 is not None:
#             kernel_params["coef0"] = self.coef0
#         kernel_params.update(self.kwargs)

#         ker_x = pairwise_kernels(X_aug, self.X_fit_, metric=self.kernel, filter_params=True, **kernel_params)

#         # Project onto the learned subspace
#         X_transformed = np.dot(ker_x, self.U[:, : self.n_components])

#         return to_backend(X_transformed, backend, reference=X_input)

#     def fit_transform(self, X, y=None, covariates=None, group_labels=None):
#         """Fit the model and transform data in one step.

#         Parameters
#         ----------
#         X : array-like
#             Input data, shape (n_samples, n_features)
#         y : array-like, optional
#             Labels, shape (n_samples,). Default is None.
#         covariates : array-like, optional
#             Domain covariates, shape (n_samples, n_covariates).
#             Alias for group_labels for backward compatibility.
#         group_labels : array-like, optional
#             Categorical variables representing domain or grouping factors,
#             shape (n_samples, n_factors). Default is None.

#         Returns
#         -------
#         X_transformed : array-like
#             Transformed data, shape (n_samples, n_components)
#         """
#         # Support both parameter names
#         factors = group_labels if group_labels is not None else covariates

#         self.fit(X, y, covariates=factors, group_labels=None)
#         return self.transform(X, covariates=factors, group_labels=None)


class MIDA(BaseKernelDomainAdapter):
    """Maximum Independent Domain Adaptation (MIDA).
    A kernel-based domain adaptation method that uses removes the effect of
    factors/covariates from the data by learning a feature space derived from maximizing
    Hilbert-Schmidt independence criterion (HSIC).

    To prevent label leakage, please set the label for the target indices to -1.

    Args:
        num_components (int, optional): Number of components to keep. If None, all components are kept.
        mu (float, optional): L2 kernel regularization coefficient. Default is 1.0.
        eta (float, optional): Class-dependency regularization coefficient. Default is 1.0.
        ignore_y (bool, optional): Whether to ignore the target variable `y` during fitting. Default is False.
        augment (str, optional): Whether to augment the input data with factors. Can be "pre" (prepend factors),
            "post" (append factors), or None (no augmentation). Defaults to None.
        kernel (str, optional): Kernel type to be used. Default is 'linear'.
        gamma (float, optional): Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Default is None.
        degree (int, optional): Degree of the polynomial kernel. Default is 3.
        coef0 (float, optional): Independent term in the polynomial and sigmoid kernels. Default is 1.
        kernel_params (dict, optional): Additional kernel parameters. Default is None.
        alpha (float, optional): Regularization parameter. Default is 1.0.
        fit_inverse_transform (bool, optional): Whether to fit the inverse transform. Default is False.
        eigen_solver (str, optional): Eigendecomposition solver to use. Default is 'auto'.
        tol (float, optional): Tolerance for convergence. Default is 0.
        max_iter (int, optional): Maximum number of iterations for the solver. Default is None.
        iterated_power (int or str, optional): Number of iterations for randomized solver. Default is 'auto'.
        remove_zero_eig (bool, optional): Whether to remove zero eigenvalues. Default is False.
        scale_components (bool, optional): Whether to scale the components. Default is False.
        random_state (int or np.random.RandomState, optional): Random seed for reproducibility. Default is None.
        copy (bool, optional): Whether to copy the input data. Default is True.
        num_jobs (int, optional): Number of jobs to run in parallel for joblib.Parallel. Default is None.
    References:
        [1] Yan, K., Kou, L. and Zhang, D., 2018. Learning domain-invariant subspace using domain features and
            independence maximization. IEEE transactions on cybernetics, 48(1), pp.288-299.
    Examples:
        >>> import numpy as np
        >>> from kale.embed.factorization import MIDA
        >>> # Generate random synthetic data
        >>> x_source = np.random.normal(loc=5, scale=1, size=(20, 40))
        >>> x_target = np.random.normal(loc=-5, scale=1, size=(20, 40))
        >>> y = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
        >>> # Concatenate source and target data
        >>> x = np.vstack((x_source, x_target))
        >>> target_indices = np.arange(20, 40)
        >>> # Mask the target indices with -1
        >>> y[target_indices] = -1
        >>> # Create factors (e.g., one-hot encoded domain labels)
        >>> factors = np.concatenate((np.zeros((20, 1)), np.ones((20, 1))), axis=0)
        >>> mida = MIDA()
        >>> x_projected = mida.fit_transform(x, y, group_labels=factors)
        >>> x_projected.shape
        (40, 18)
    """

    _parameter_constraints: dict = {
        **BaseKernelDomainAdapter._parameter_constraints,
        "mu": [Interval(Real, 0, None, closed="neither")],
        "eta": [Interval(Real, 0, None, closed="neither")],
    }

    def __init__(
        self,
        num_components=None,
        mu=1.0,
        eta=1.0,
        ignore_y=False,
        augment=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
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
        # L2 kernel regularization parameter
        self.mu = mu
        # Class dependency regularization parameter
        self.eta = eta

        # Kernel and Eigendecomposition parameters
        super().__init__(
            num_components=num_components,
            ignore_y=ignore_y,
            augment=augment,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
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
            num_jobs=num_jobs,
        )

    def _make_objective_kernel(self, k_x, y, group_labels):
        # equivalent to `H` in the original paper
        h = _centering_kernel(_num_features(k_x), k_x.dtype)
        # linear kernel used for the label and factors
        k_y = pairwise_kernels(y, n_jobs=self.num_jobs)
        k_f = pairwise_kernels(group_labels, n_jobs=self.num_jobs)

        centerer = KernelCenterer()
        k_y = centerer.fit_transform(k_y)
        k_f = centerer.fit_transform(k_f)

        k_objective = multi_dot((k_x, self.mu * h + self.eta * k_y - k_f, k_x))

        return k_objective
