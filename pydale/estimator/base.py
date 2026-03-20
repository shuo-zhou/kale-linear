import warnings

import numpy as np
import osqp
import scipy.sparse as sparse
from cvxopt import matrix, solvers
from numpy.linalg import inv, multi_dot
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted


class BaseFramework(BaseEstimator, ClassifierMixin):
    """Semi-supervised Learning Framework
    """

    def __init__(self, kernel, k_neighbour=5, manifold_metric="cosine", knn_mode="distance", **kwargs) -> None:
        super().__init__()
        self.kernel = kernel
        self.k_neighbour = k_neighbour
        self.manifold_metric = manifold_metric
        self.knn_mode = knn_mode
        self.coef_ = None
        self._lb = LabelBinarizer(pos_label=1, neg_label=-1)
        self.kwargs = kwargs
        self.x = None

    @classmethod
    def _solve_semi_dual(cls, krnl_x, y, obj_core, C, solver="osqp"):
        """[summary]

        Parameters
        ----------
        krnl_x : [type]
            [description]
        y : [type]
            [description]
        obj_core : [array-like]
            Core matrix of objective
        C : [float]
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be
            strictly positive. The penalty is a squared l2 penalty.
        solver : str, optional
            [description], by default 'osqp'

        Returns
        -------
        [type]
            [description]
        """
        if len(y.shape) == 1:
            coef_, support_ = cls._semi_binary_dual(krnl_x, y, obj_core, C, solver)
            support_ = [support_]
        else:
            coef_ = np.zeros((krnl_x.shape[1], y.shape[1]))
            support_ = []
            for i in range(y.shape[1]):
                coef_i, support_i = cls._semi_binary_dual(krnl_x, y[:, i], obj_core, C, solver)
                coef_[:, i] = coef_i
                support_.append(support_i)

        return coef_, support_

    @classmethod
    def _semi_binary_dual(cls, krnl_x, y, obj_core, C, solver="osqp"):
        """solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b

        Parameters
        ----------
        krnl_x : [type]
            [description]
        y : [type]
            [description]
        obj_core : [type]
            [description]
        C : [float]
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be
            strictly positive. The penalty is a squared l2 penalty.
        solver : str, optional
            [description], by default 'osqp'

        Returns
        -------
        [type]
            [description]
        """
        n_labeled = y.shape[0]
        n = krnl_x.shape[0]
        J = np.zeros((n_labeled, n))
        J[:n_labeled, :n_labeled] = np.eye(n_labeled)
        obj_inv = inv(obj_core)
        y_diag = np.diag(y.reshape(-1))
        Q = multi_dot([y_diag, J, krnl_x, obj_inv, J.T, y_diag])
        Q = Q.astype("float32")
        alpha = cls._quadprog(Q, y, C, solver)
        coef_ = multi_dot([obj_inv, J.T, y_diag, alpha])
        support_ = np.where((alpha > 0) & (alpha < C))
        return coef_, support_

    @classmethod
    def _quadprog(cls, P, y, C, solver="osqp"):
        """solve min_x x^TPx + q^Tx, s.t. Gx<=h, Ax=b

        Parameters
        ----------
        P : [type]
            [description]
        y : [type]
            [description]
        C : [float]
            Regularization parameter. The strength of the regularization is inversely proportional to C. Must be
            strictly positive. The penalty is a squared l2 penalty.
        solver : str, optional
            quadprog solver name, by default 'osqp'

        Returns
        -------
        array-like
            coefficients beta
        """
        # dual
        n_labeled = y.shape[0]
        q = -1 * np.ones((n_labeled, 1))

        if solver == "cvxopt":
            G = np.zeros((2 * n_labeled, n_labeled))
            G[:n_labeled, :] = -1 * np.eye(n_labeled)
            G[n_labeled:, :] = np.eye(n_labeled)
            h = np.zeros((2 * n_labeled, 1))
            h[n_labeled:, :] = C / n_labeled

            # convert numpy matrix to cvxopt matrix
            P = matrix(P.astype(np.float))
            q = matrix(q)
            G = matrix(G)
            h = matrix(h)
            A = matrix(y.reshape(1, -1).astype("float64"))
            b = matrix(np.zeros(1).astype("float64"))

            solvers.options["show_progress"] = False
            sol = solvers.qp(P, q, G, h, A, b)

            beta = np.array(sol["x"]).reshape(n_labeled)

        elif solver == "osqp":
            warnings.simplefilter("ignore", sparse.SparseEfficiencyWarning)
            P_sparse = sparse.csc_matrix((n_labeled, n_labeled))
            P_sparse[:n_labeled, :n_labeled] = P[:n_labeled, :n_labeled]
            G = sparse.vstack([sparse.eye(n_labeled), y.reshape(1, -1)]).tocsc()
            l_ = np.zeros((n_labeled + 1, 1))
            u = np.zeros(l_.shape)
            u[:n_labeled, 0] = C / n_labeled

            prob = osqp.OSQP()
            prob.setup(P_sparse, q, G, l_, u, verbose=False)
            res = prob.solve()
            beta = res.x

        else:
            raise ValueError("Invalid QP solver")

        return beta

    @classmethod
    def _solve_semi_ls(cls, Q, y):
        """[summary]

        Parameters
        ----------
        Q : [type]
            [description]
        y : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        """
        n = Q.shape[0]
        n_labeled = y.shape[0]
        Q_inv = inv(Q)
        if len(y.shape) == 1:
            y_ = np.zeros(n)
            y_[:n_labeled] = y[:]
        else:
            y_ = np.zeros((n, y.shape[1]))
            y_[:n_labeled, :] = y[:, :]
        return np.dot(Q_inv, y_)

    def predict(self, x):
        """Perform classification on samples in x.

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)

        Returns
        -------
        array-like
            predicted labels, shape (n_samples,)
        """
        dec_scores = self.decision_function(x)
        # if self._lb.y_type_ == 'binary':
        #     y_pred = self._lb.inverse_transform(np.sign(dec_scores).reshape(-1, 1))
        # else:
        #     y_pred = self._lb.inverse_transform(dec_scores)
        y_pred = self._lb.inverse_transform(dec_scores)

        return y_pred

    def decision_function(self, x):
        """Evaluates the decision function for the samples in x

        Parameters
        ----------
        x : array-like
            Input data, shape (n_samples, n_features)

        Returns
        -------
        array-like
            decision scores, shape (n_samples,) for binary classification,
            (n_samples, n_class) for multi-class cases
        """
        check_is_fitted(self, "x")
        krnl_x = pairwise_kernels(x, self.x, metric=self.kernel, filter_params=True, **self.kwargs)
        return np.dot(krnl_x, self.coef_)
