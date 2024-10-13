#adagrad : vt = vt-1 + (delW)**2

import numpy as np
import matplotlib.pyplot as plt
import math

X = [0.5, 1.0, 2.5]
Y = [0.2, 0.84, 0.99]


def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))


def error(w, b):
    err = 0
    for x, y in zip(X, Y):
        fx = f(x, w, b)
        err += (fx - y) ** 2
    return err


def grad_w(x, y, w, b):
    fx = f(x, w, b)
    return (fx - y) * (1 - fx) * fx * x


def grad_b(x, y, w, b):
    fx = f(x, w, b)
    return (fx - y) * (1 - fx) * fx


def do_adagrad():
    max_epochs = 1000
    w, b, eta, eps, ub, uw = -2, -2, 1.0, 0.00000001, 0.0, 0.0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, y, w, b)
            db += grad_b(x, y, w, b)
        uw = uw + dw ** 2
        ub = ub + db ** 2
        w = w - eta * dw / (np.sqrt(uw) + eps)
        b = b - eta * db / (np.sqrt(ub) + eps)

        current_error = error(w, b)
        errors.append(current_error)
        epochs.append(i)

        if i == 999:
            y_pred = [f(x, w, b) for x in X]
            plt.plot(X, Y, 'ro', label='True Y')
            plt.plot(X, Y, 'b-', label='Predicted Y')
            plt.title(f'Epoch {i}')
            plt.legend()
            plt.show()

    plt.plot(epochs, errors)
    plt.title('Error over epochs_adagrad')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    do_adagrad()