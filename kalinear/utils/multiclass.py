import numpy as np


def score2pred(scores):
    """Convert decision scores to a one-vs-rest label matrix."""
    n_samples, n_classes = scores.shape
    y_pred_ = -1 * np.ones((n_samples, n_classes))
    dec_sort = np.argsort(scores, axis=1)[:, ::-1]
    for i in range(n_samples):
        y_pred_[i, dec_sort[i, 0]] = 1

    return y_pred_
