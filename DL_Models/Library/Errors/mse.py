import numpy as np

def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))


def error(w, b, X, Y):
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(x, w, b)
        err += (fx - y) ** 2
    return 0.5 * err
