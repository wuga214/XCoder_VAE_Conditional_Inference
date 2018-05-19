import numpy as np
from functools import wraps

def maximum_mean_discrepancy(x, y):
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    subtract = x_mean-y_mean
    return np.linalg.norm(subtract)

def compareto(y=None):
    def _my_decorator(func):
        def _decorator(*args, **kwargs):
            response = func(*args, y=y)
            return response
        return wraps(func)(_decorator)
    return _my_decorator
