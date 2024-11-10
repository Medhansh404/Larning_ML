import numpy as np


def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))


def grad_b(x, w, b, y):
    fx = f(x, w, b)
    return (fx - y) * fx * (1 - fx)


def grad_w(x, w, b, y):
    fx = f(x, w, b)
    return (fx - y) * fx * (1 - fx) * x
