import numpy as np
from numpy.lib.utils import source


def differentiate(expression, x):
    gradient = np.zeros_like(x)
    x_iter = np.nditer(x, flags=['multi_index'])

    while not x_iter.finished:
        mi = x_iter.multi_index
        source = x[mi]
        limited_distance = 1e-4 * source
        y = expression(x)
        x[mi] += limited_distance
        y_plus_dx = expression(x)
        x[mi] = source
        temp = (y_plus_dx - y) / (limited_distance)
        gradient[mi] = temp
        x_iter.iternext()
        
    return gradient