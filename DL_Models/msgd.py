import numpy as np
import matplotlib.pyplot as plt

X = [0.5, 1.0, 2.5]
Y = [0.2, 0.84, 0.9]


def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))


def error(w, b):
    err = 0
    for x,y in zip(X,Y):
        fx = f(x, w, b)
        err += (fx - y) ** 2
    return 0.5 * err


def grad_w(x, y, w, b):
    fx = f(x, w, b)
    return (fx - y) * (1 - fx) * fx * x


def grad_b(x, y, w, b):
    fx = f(x, w, b)
    return (fx - y) * (1 - fx) * fx


def do_msgd():
    max_epochs = 1000
    w, b, beta, eta = -2, -2, 0.9, 1.0
    prev_wu, prev_bu = 0, 0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw = grad_w(x, y, w, b)
            db = grad_b(x, y, w, b)
            uw = prev_wu * beta + eta * dw
            ub = prev_bu * beta + eta * db
            w = w - uw
            b = b - ub
            prev_wu = uw
            prev_bu = ub

        current_error = error(w, b)
        errors.append(current_error)
        epochs.append(i)

        if i == 999:
            Y_pred = [f(x, w, b) for x in X]
            plt.plot(X, Y, 'ro', label='True Y')
            plt.plot(X, Y_pred, 'b-', label='Predicted Y')
            plt.title(f'Epoch {i}')
            plt.legend()
            plt.show()

            plt.plot(epochs, errors)
            plt.title('Error over epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.show()


if __name__ == "__main__":
    do_msgd()
