from time import time

import numpy as np

# import torch
from numpy.linalg import multi_dot
from scipy.special import expit
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from kalelinear._covariates import check_numeric_covariates, fit_covariate_encoder

from .base import BaseKaleEstimator


def simple_hsic_grad_term(w, X, groups):
    """Compute the simplified HSIC gradient term.
    Simplified HSIC is defined as w^T X^T H G G^T H X w, where H is the centering matrix and G is the group indicator matrix.
    This function computes X^T H G G^T H X w, which is the part of the gradient that depends on the data, groups, and the model parameters w.

    Parameters
    ----------
    w : array-like of shape (n_features,)
        Model coefficients.
    X : array-like of shape (n_samples, n_features)
        Input data.
    groups : array-like of shape (n_samples, n_groups)
        Group or domain indicators.

    Returns
    -------
    array-like, shape (n_features,)
        Simplified HSIC gradient term.
    """
    n = X.shape[0]
    centering_matrix = np.eye(n) - np.ones((n, n)) / n

    return multi_dot((X.T, centering_matrix, groups, groups.T, centering_matrix, X, w))


def _compute_pred_loss(y, y_hat):
    """Compute the binary prediction log loss.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        True binary labels.
    y_hat : array-like of shape (n_samples,)
        Predicted probabilities.

    Returns
    -------
    float
        Log loss value.
    """
    pred_log_loss = -1 * (np.dot(y, np.log(y_hat + 1e-6)) + np.dot((1 - y), np.log(1 - y_hat + 1e-6))) / y.shape[0]
    return pred_log_loss


class GSDA(BaseKaleEstimator):
    """Group-specific logistic classifier with HSIC regularization.

    Parameters
    ----------
    lr : int or float, default=0.1
        Learning rate used by gradient-based optimizers.
    max_iter : int, default=100
        Maximum number of optimization iterations.
    regularization : {None, "l2"}, default="l2"
        Optional regularization strategy for the prediction objective.
    l2_hparam : float, default=1.0
        L2 regularization strength multiplier.
    tolerance_grad : float, default=1e-7
        Stopping threshold for gradient magnitude.
    tolerance_change : float, default=1e-9
        Stopping threshold for objective change.
    lambda_ : float, default=1.0
        Weight for the HSIC-based penalty term.
    optimizer : {"gd", "lbfgs"}, default="gd"
        Optimization algorithm used in :meth:`fit`.
    memory_size : int, default=10
        History length used by the L-BFGS approximation.
    random_state : int or None, default=None
        Random seed used for initialization.

    Attributes
    ----------
    theta_ : ndarray of shape (n_features + 1,)
        Model parameters where the first entry is the intercept.
    losses : dict
        Optimization history for objective, prediction, HSIC loss, and runtime.
    """

    def __init__(
        self,
        lr=0.1,
        max_iter=100,
        regularization="l2",
        l2_hparam=1.0,
        tolerance_grad=1e-07,
        tolerance_change=1e-9,
        lambda_=1.0,
        optimizer="gd",
        memory_size=10,
        random_state=None,
        covariate_encoder=None,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.regularization = regularization
        self.alpha = l2_hparam
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.theta_ = None
        self.lambda_ = lambda_
        self.losses = {"ovr": [], "pred": [], "hsic": [], "time": []}
        self.optimizer = optimizer
        self.memory_size = memory_size
        self.random_state = random_state
        self.covariate_encoder = covariate_encoder

    def fit(self, X, y, groups, target_idx=None):
        """Fit the GSDA classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Binary labels for optimization.
        groups : array-like of shape (n_samples,) or (n_samples, n_groups)
            Group/domain indicators used by the HSIC term.
        target_idx : array-like of shape (n_target_samples,), optional
            Indices indicating target samples. If ``None``, the first ``len(y)`` rows are used.

        Returns
        -------
        self : GSDA
            Fitted estimator.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        y = np.asarray(y)
        groups, encoder = fit_covariate_encoder(
            groups,
            self.covariate_encoder,
            n_samples,
            error_prefix="Groups",
        )
        self.covariate_encoder_ = encoder
        groups = check_numeric_covariates(
            groups,
            n_samples,
            allow_none=False,
            error_prefix="Groups",
            numeric_error_message=(
                "Groups must be numeric when `covariate_encoder` is None. "
                "Provide numeric groups or set `covariate_encoder`."
            ),
        )
        # ensure X, y, and groups have compatible shapes
        if n_samples < y.shape[0]:
            raise ValueError("Mismatched number of samples between X, y, and groups.")
        if isinstance(target_idx, (list, np.ndarray)) and len(target_idx) > y.shape[0]:
            raise ValueError("Length of target_idx cannot exceed number of target samples.")
        existing_losses = getattr(self, "losses", None)
        if isinstance(existing_losses, dict):
            self.losses = {key: [] for key in existing_losses}
        else:
            self.losses = {"ovr": [], "pred": [], "hsic": [], "time": []}
        rng = np.random if self.random_state is None else np.random.RandomState(self.random_state)
        self.theta_ = rng.random(
            (X.shape[1] + 1),
        )
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        start_time = time()
        if self.optimizer == "lbfgs":
            self._lbfgs_solver(X, y, groups, target_idx)
        else:
            self._gd_solver(X, y, groups, target_idx)
        time_used = time() - start_time
        self.losses["time"].append(time_used)

        return self

    def predict_proba(self, X):
        """Estimate class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        probs : ndarray of shape (n_samples,)
            Positive-class probabilities.
        """
        check_is_fitted(self)
        if getattr(self, "theta_", None) is None:
            raise NotFittedError("This estimator is not fitted yet. Call 'fit' before using this estimator.")

        return expit((X @ self.theta_[1:]) + self.theta_[0])

    def predict(self, X):
        """Predict binary class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class label per sample.
        """
        y_proba = self.predict_proba(X)
        y_pred = np.zeros(y_proba.shape)
        y_pred[np.where(y_proba > 0.5)] = 1

        return y_pred

    @property
    def intercept_(self):
        """Fitted intercept term.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, "theta_") or self.theta_ is None:
            raise AttributeError("This GSDA instance is not fitted yet. Call 'fit' before accessing 'intercept_'.")
        return self.theta_[0]

    @property
    def coef_(self):
        """Fitted coefficients.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, "theta_") or self.theta_ is None:
            raise AttributeError("This GSDA instance is not fitted yet. Call 'fit' before accessing 'coef_'.")
        return self.theta_[1:]

    def get_fitted_params(self):
        """Return fitted coefficients and intercept.

        Returns
        -------
        params : dict
            Dictionary with ``intercept`` and ``coef`` keys.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, "theta_") or self.theta_ is None:
            raise RuntimeError("This GSDA instance is not fitted yet. Call 'fit' before requesting fitted parameters.")
        params = dict()
        params["intercept"] = self.theta_[0]
        params["coef"] = self.theta_[1:]
        return params

    def _lbfgs_solver(self, X, y, groups, target_idx=None):
        """Optimize parameters using a limited-memory BFGS-like update.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1)
            Augmented design matrix including the intercept column.
        y : ndarray of shape (n_target,)
            Target labels.
        groups : ndarray of shape (n_samples, n_groups)
            Group indicators.
        target_idx : ndarray of shape (n_target,), optional
            Indices for target samples.

        Returns
        -------
        self : GSDA
            Estimator with updated ``theta_``.
        """
        delta_theta = []  # Δx
        delta_grads = []  # Δgrad

        grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(X, y, groups, target_idx)
        theta_old = self.theta_.copy()
        grad_old = grad.copy()

        for _ in range(self.max_iter):
            q = grad.copy()
            # Compute the search direction
            if _ > 0:
                alphas = []
                for i in reversed(range(len(delta_theta))):
                    alpha = delta_theta[i].dot(q) / delta_grads[i].dot(delta_theta[i])
                    alphas.append(alpha)
                    q -= alpha * delta_grads[i]

                gamma_k = delta_theta[-1].dot(delta_grads[-1]) / delta_grads[-1].dot(delta_grads[-1])

                z = (gamma_k * np.eye(self.theta_.shape[0])) @ q

                for i in range(len(delta_theta)):
                    beta_i = delta_grads[i].dot(z) / delta_grads[i].dot(delta_theta[i])
                    z += delta_theta[i] * (alphas[len(alphas) - 1 - i] - beta_i)
            else:
                z = np.eye(self.theta_.shape[0]) @ q
            # Line search and update x
            # Implement a line search algorithm to find an appropriate step size
            # step_size = 1  # Placeholder
            # Update memory
            if len(delta_theta) == self.memory_size:
                delta_theta.pop(0)
                delta_grads.pop(0)

            if _ > 0 and _ % 10 == 0 and self.lr > 0.001:
                self.lr *= 0.8

            self.theta_ = theta_old - z * self.lr

            delta_theta.append(self.theta_ - theta_old)
            theta_old = self.theta_.copy()

            # Update gradient
            grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(X, y, groups, target_idx)

            self.losses["ovr"].append(pred_log_loss + hsic_log_loss)
            self.losses["pred"].append(pred_log_loss)
            self.losses["hsic"].append(hsic_log_loss)

            delta_grads.append(grad - grad_old)
            grad_old = grad.copy()

            if _ > self.memory_size * 2 and (self._terminate_grad(delta_grads[-1]) or self._terminate_change()):
                break

        return self

    def _gd_solver(self, X, y, groups, target_idx=None):
        """Optimize parameters using gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features + 1)
            Augmented design matrix including the intercept column.
        y : ndarray of shape (n_target,)
            Target labels.
        groups : ndarray of shape (n_samples, n_groups)
            Group indicators.
        target_idx : ndarray of shape (n_target,), optional
            Indices for target samples.

        Returns
        -------
        self : GSDA
            Estimator with updated ``theta_``.
        """
        for _ in range(self.max_iter):
            delta_grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(X, y, groups, target_idx)
            if _ % 10 == 0:
                self.losses["ovr"].append(pred_log_loss + hsic_log_loss)
                self.losses["pred"].append(pred_log_loss)
                self.losses["hsic"].append(hsic_log_loss)
                # if np.abs(self.losses["ovr"][-1] - self.losses["ovr"][-2]) < self.tolerance:
                #     break
                if len(self.losses["ovr"]) > 10 and self._terminate_change():
                    break

            if _ % 50 == 0 and self.lr > 0.001:
                self.lr *= 0.8

            if not self._terminate_grad(delta_grad):
                self.theta_ -= self.lr * delta_grad
            else:
                break

        return self

    def compute_gsda_gradient(self, X, y, groups, target_idx=None):
        n_sample = X.shape[0]
        n_tgt = y.shape[0]
        if target_idx is None:
            x_tgt = X[:n_tgt]
        else:
            x_tgt = X[target_idx]

        y_hat = expit(x_tgt @ self.theta_)
        # n_feature = X.shape[1]
        _simple_hsic = simple_hsic_grad_term(self.theta_, X, groups)
        hsic_proba = expit(multi_dot((self.theta_, _simple_hsic)) / np.square(n_sample - 1))
        grad_hsic = (hsic_proba - 1) * _simple_hsic / np.square(n_sample - 1)

        delta_grad = (x_tgt.T @ (y_hat - y)) / n_tgt
        if self.regularization is not None:
            delta_grad += self.theta_ * self.alpha
        delta_grad += self.lambda_ * grad_hsic

        hsic_log_loss = -1 * np.log(hsic_proba)
        pred_log_loss = _compute_pred_loss(y, y_hat)

        return delta_grad, pred_log_loss, hsic_log_loss

    def _terminate_change(self):
        if (
            self.losses["ovr"][-1] > self.losses["ovr"][-2]
            or self.losses["ovr"][-2] - self.losses["ovr"][-1] < self.tolerance_change
        ):
            return True
        else:
            return False

    def _terminate_grad(self, delta_grad):
        return np.all(abs(delta_grad) <= self.tolerance_grad)
