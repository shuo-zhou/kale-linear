import numpy as np
from numpy.linalg import eig, inv, multi_dot
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer, LabelBinarizer
from sklearn.utils.validation import check_is_fitted


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, kernel="linear", **kwargs) -> None:
        super().__init__()
        self.n_components = n_components
        self.kernel = kernel
        self.kwargs = kwargs

        self.U = None
        self.x_fit = None
        self._lb = LabelBinarizer(pos_label=1, neg_label=0)
        self._centerer = KernelCenterer()

    def _fit(self, obj_min, obj_max):
        """
        solve eigen-decomposition

        Parameters
        ----------
        obj_min : array-like,
            objective matrix to minimise, shape (n_samples, n_features)
        obj_max : array-like,
            objective matrix to maximise, shape (n_samples, n_features)
        Returns
        -------
        self
        """
        obj_ovr = np.dot(inv(obj_min), obj_max)
        eig_values, eig_vectors = eig(obj_ovr)
        idx_sorted = eig_values.argsort()[::-1]

        self.U = eig_vectors[:, idx_sorted]
        self.U = np.asarray(self.U, dtype=np.float)

        return self

    def transform(self, x, aug_features=None):
        """
        Parameters
        ----------
        x : array-like,
            shape (n_samples, n_features)
        aug_features : array-like,
            Augmentation features, shape (n_samples, n_aug_features)
        Returns
        -------
        array-like
            transformed data
        """
        check_is_fitted(self, "x_fit")
        if type(aug_features) == np.ndarray:
            x = np.concatenate((x, aug_features), axis=1)
        krnl_x = self._centerer.transform(
            pairwise_kernels(x, self.x_fit, metric=self.kernel, filter_params=True, **self.kwargs)
        )

        return np.dot(krnl_x, self.U[:, : self.n_components])
