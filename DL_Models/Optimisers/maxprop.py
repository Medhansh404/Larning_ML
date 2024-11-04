import numpy as np
import matplotlib.pyplot as plt

X = [0.2, 0.5, 0.9]
Y = [0.2, 0.84, 0.99]


def f(x, w, b):
    return 1 / (1+ np.exp(-(w*x +b)))


def error(w, b):
    err = 0
    for (x,y) in zip(X, Y):
        fx = f(x, w, b)
        err += (fx-y) **2
    return err


def grad_w(x, y, w, b):
    fx = f(x, y, w, b)
    return (fx-y) * (1-x) * x * fx


def grad_b(x, y, w, b):
    fx = f(x, y, w, b)
    return (fx-y) * (1-x) * fx


def do_maxprop():