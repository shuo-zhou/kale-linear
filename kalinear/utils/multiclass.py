import numpy as np


def score2pred(scores):
    """
    Converting decision scores (probability) to predictions
    Parameter:
        scores: score matrix, array-like, shape (n_samples, n_class)
    Return:
        prediction matrix (1, -1), array-like, shape (n_samples, n_class)
    """
    n = scores.shape[0]
    y_pred_ = -1 * np.ones((n, n))
    dec_sort = np.argsort(scores, axis=1)[:, ::-1]
    for i in range(n):
        label_idx = dec_sort[i, 0]
        y_pred_[i, label_idx] = 1

    return y_pred_
