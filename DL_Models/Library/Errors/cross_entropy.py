import numpy as np

X = []
Y = []


def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))

def error(w, b):
    err = 0.0
    for x, y in zip(X, Y):
        loss = -(f(x) * y + (1-f(x) * (1-y))