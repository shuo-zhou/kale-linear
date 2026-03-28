from time import time

import numpy as np

# import torch
from numpy.linalg import multi_dot
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


def simple_hsic(w, x, groups):
    n = x.shape[0]
    ctr_mat = np.diag(np.ones(n)) - 1 / n

    return multi_dot((x.T, ctr_mat, groups, groups.T, ctr_mat, x, w))


def _compute_pred_loss(y, y_hat):
    pred_log_loss = -1 * (np.dot(y, np.log(y_hat + 1e-6)) + np.dot((1 - y), np.log(1 - y_hat + 1e-6))) / y.shape[0]
    return pred_log_loss


class GSDA(BaseEstimator, ClassifierMixin):
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
    theta : ndarray of shape (n_features + 1,)
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
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.regularization = regularization
        self.alpha = l2_hparam
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.theta = None
        self.lambda_ = lambda_
        self.losses = {"ovr": [], "pred": [], "hsic": [], "time": []}
        self.optimizer = optimizer
        self.memory_size = memory_size
        self.random_state = random_state

    def fit(self, x, y, groups, target_idx=None):
        """Fit the GSDA classifier.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Binary labels for optimization.
        groups : array-like of shape (n_samples,) or (n_samples, n_groups)
            Group/domain indicators used by the HSIC term.
        target_idx : array-like of shape (n_target,), optional
            Indices indicating target samples. If ``None``, the first ``len(y)`` rows are used.

        Returns
        -------
        self : GSDA
            Fitted estimator.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        groups = np.asarray(groups)
        if groups.ndim == 1:
            groups = groups.reshape((-1, 1))
        self.theta = np.random.random(
            (x.shape[1] + 1),
        )
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

        start_time = time()
        if self.optimizer == "lbfgs":
            self._lbfgs_solver(x, y, groups, target_idx)
        else:
            self._gd_solver(x, y, groups, target_idx)
        time_used = time() - start_time
        self.losses["time"].append(time_used)

        return self

    def predict_proba(self, x):
        """Estimate class probabilities.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        probs : ndarray of shape (n_samples,)
            Positive-class probabilities.
        """
        return expit((x @ self.theta[1:]) + self.theta[0])

    def predict(self, x):
        """Predict binary class labels.

        Parameters
        ----------
        x : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class label per sample.
        """
        y_proba = self.predict_proba(x)
        y_pred = np.zeros(y_proba.shape)
        y_pred[np.where(y_proba > 0.5)] = 1

        return y_pred

    def get_params(self, **kwargs):
        """Return fitted coefficients and intercept.

        Returns
        -------
        params : dict
            Dictionary with ``intercept`` and ``coef`` keys.
        """
        try:
            params = dict()
            params["intercept"] = self.theta[0]
            params["coef"] = self.theta[1:]
            return params
        except self.theta is None:
            raise Exception("Fit the model first!")

    def _lbfgs_solver(self, x, y, groups, target_idx=None):
        """Optimize parameters using a limited-memory BFGS-like update.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features + 1)
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
            Estimator with updated ``theta``.
        """
        delta_theta = []  # Δx
        delta_grads = []  # Δgrad

        grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(x, y, groups, target_idx)
        theta_old = self.theta.copy()
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

                z = (gamma_k * np.eye(self.theta.shape[0])) @ q

                for i in range(len(delta_theta)):
                    beta_i = delta_grads[i].dot(z) / delta_grads[i].dot(delta_theta[i])
                    z += delta_theta[i] * (alphas[len(alphas) - 1 - i] - beta_i)
            else:
                z = np.eye(self.theta.shape[0]) @ q
            # Line search and update x
            # Implement a line search algorithm to find an appropriate step size
            # step_size = 1  # Placeholder
            # Update memory
            if len(delta_theta) == self.memory_size:
                delta_theta.pop(0)
                delta_grads.pop(0)

            if _ > 0 and _ % 10 == 0 and self.lr > 0.001:
                self.lr *= 0.8

            self.theta = theta_old - z * self.lr

            delta_theta.append(self.theta - theta_old)
            theta_old = self.theta.copy()

            # Update gradient
            grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(x, y, groups, target_idx)

            self.losses["ovr"].append(pred_log_loss + hsic_log_loss)
            self.losses["pred"].append(pred_log_loss)
            self.losses["hsic"].append(hsic_log_loss)

            delta_grads.append(grad - grad_old)
            grad_old = grad.copy()

            if _ > self.memory_size * 2 and (self._terminate_grad(delta_grads[-1]) or self._terminate_change()):
                break

        return self

    def _gd_solver(self, x, y, groups, target_idx=None):
        """Optimize parameters using gradient descent.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features + 1)
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
            Estimator with updated ``theta``.
        """
        for _ in range(self.max_iter):
            delta_grad, pred_log_loss, hsic_log_loss = self.compute_gsda_gradient(x, y, groups, target_idx)
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
                self.theta -= self.lr * delta_grad
            else:
                break

        return self

    def compute_gsda_gradient(self, x, y, groups, target_idx=None):
        n_sample = x.shape[0]
        n_tgt = y.shape[0]
        if target_idx is None:
            x_tgt = x[:n_tgt]
        else:
            x_tgt = x[target_idx]

        y_hat = expit(x_tgt @ self.theta)
        # n_feature = x.shape[1]
        _simple_hsic = simple_hsic(self.theta, x, groups)
        hsic_proba = expit(multi_dot((self.theta, _simple_hsic)) / np.square(n_sample - 1))
        grad_hsic = (hsic_proba - 1) * _simple_hsic / np.square(n_sample - 1)

        if self.regularization is not None:
            delta_grad = (x_tgt.T @ (y_hat - y)) / n_tgt + self.theta * self.alpha + self.lambda_ * grad_hsic
        else:
            delta_grad = x_tgt.T @ (y_hat - y)

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
