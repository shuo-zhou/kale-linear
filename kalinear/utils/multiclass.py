import numpy as np

from ._backend import infer_backend, to_backend, to_numpy


def score2pred(scores):
    """Convert decision scores to a one-vs-rest label matrix."""
    backend = infer_backend(scores)
    scores_np = to_numpy(scores)
    n_samples, n_classes = scores_np.shape
    y_pred_ = -1 * np.ones((n_samples, n_classes))
    dec_sort = np.argsort(scores_np, axis=1)[:, ::-1]
    for i in range(n_samples):
        y_pred_[i, dec_sort[i, 0]] = 1

    return to_backend(y_pred_, backend, reference=scores)
