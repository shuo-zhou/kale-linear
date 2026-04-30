import numpy as np


def to_numpy(value):
    if value is None:
        return None
    return np.asarray(value)
