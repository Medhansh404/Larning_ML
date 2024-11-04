import numpy as np
import matplotlib.pyplot as plt
X = [0.5, 1.0,  2.5]
Y = [0.2, 0.84, 0.9]


def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))


def error(w, b):
    err = 0.0
    for x, y in zip(X, Y):
        fx = f(x, w, b)
        err += (fx - y) ** 2
    return 0.5 * err


def grad_b(x, w, b, y):
    fx = f(x, w, b)
    return (fx - y) * fx * (1 - fx)


def grad_w(x, w, b, y):
    fx = f(x, w, b)
    return (fx - y) * fx * (1 - fx) * x


def do_mombgd():
    max_epochs = 1000
    errors, epochs = [], []

    w, b, eta = -2, -2, 1.0
    prev_uw, prev_ub, beta = 0, 0, 0.9

    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        uw = beta * prev_uw + eta * dw
        ub = beta * prev_ub + eta * db
        w = w - uw
        b = b - ub
        prev_uw = uw
        prev_ub = ub

        current_error = error(w, b)
        errors.append(current_error)
        epochs.append(i)

        if i % 100 == 0:
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
    do_mombgd()

