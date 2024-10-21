
import numpy as np
import matplotlib.pyplot as plt


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


def do_adadelta():
    max_epochs = 1000
    errors, epochs = [], []
    w, b = -4.0, -4.0
    beta = 0.99
    v_w, v_b, eps = 0, 0, 1e-4
    u_w, u_b = 0, 0

    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        v_w = beta * v_w + (1 - beta) * dw ** 2
        v_b = beta * v_b + (1 - beta) * db ** 2

        delta_w = dw * np.sqrt(u_w + eps) / (np.sqrt(v_w + eps))
        delta_b = db * np.sqrt(u_b + eps) / (np.sqrt(v_b + eps))
        u_w = beta * u_w + (1 - beta) * delta_w ** 2
        u_b = beta * u_b + (1 - beta) * delta_b ** 2

        w = w - delta_w
        b = b - delta_b

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
    plt.title('Error over epochs_adadelta')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    do_adadelta()