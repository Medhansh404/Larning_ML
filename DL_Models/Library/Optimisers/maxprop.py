import numpy as np
import matplotlib.pyplot as plt

X = [0.2, 0.5, 0.9]
Y = [0.2, 0.84, 0.99]


def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))


def error(w, b):
    err = 0
    for (x, y) in zip(X, Y):
        fx = f(x, w, b)
        err += (fx - y) ** 2
    return err


def grad_w(x, y, w, b):
    fx = f(x, w, b)
    return (fx - y) * (1 - x) * x * fx


def grad_b(x, y, w, b):
    fx = f(x,w, b)
    return (fx - y) * (1 - x) * fx


# max prop is the maximum approximation on the basis of l2 to linf normalisation
def do_maxprop():
    max_epochs = 1000
    w, b, eta, beta, eps = -2, -2, 0.1, 0.5, 0.000000001
    prev_ub, prev_uw = 0, 0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, y, w, b)
            db += grad_b(x, y, w, b)
        uw = max(beta * prev_uw, abs(dw))
        ub = max(beta * prev_ub, abs(db))

        w = w - eta * dw / (uw + eps)
        b = b - eta * db / (ub + eps)

        prev_ub = ub
        prev_uw = uw

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
    plt.title('Error over epochs_rmsprop')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    do_maxprop()
