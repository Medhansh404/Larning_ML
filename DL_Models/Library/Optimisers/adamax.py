import numpy as np
import matplotlib.pyplot as plt

X = [0.2, 0.5, 0.9]
Y = [0.2, 0.84, 0.99]


def f(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))


def error(w, b):
    err = 0
    for (x,y) in zip(X, Y):
        fx = f(x, w, b)
        err += (fx-y) ** 2
    return err


def grad_w(x, y, w, b):
    fx = f(x, w, b)
    return (fx-y) * (1-x) * x * fx


def grad_b(x, y, w, b):
    fx = f(x, w, b)
    return (fx-y) * (1-x) * fx


def do_adamax():
    max_epochs, w, b, eta, e = 1000, -2.0, -2.0, 0.1, .00000001
    beta1, beta2 = 0.1, 0.1
    errors, epochs = [], []
    prev_mt_w, prev_mt_b, prev_vt_w, prev_vt_b = 0, 0, 0, 0
    for epoch in range(max_epochs):
        dw, db = 0, 0
        for (x, y) in zip(X, Y):
            dw += grad_w(x, y, w, b)
            db += grad_b(x, y, w, b)

        mt_w = beta1 * prev_mt_w + (1 - beta1) * dw
        mt_b = beta1 * prev_mt_b + (1 - beta1) * db
        vt_w = max(beta2 * prev_vt_w, abs(dw))
        vt_b = max(beta2 * prev_vt_b, abs(db))

        mt_w /= 1 - beta1
        mt_b /= 1 - beta1

        w = w - eta * mt_w/(vt_w+e)
        b = b - eta * mt_b/(vt_b+e)

        prev_mt_w, prev_mt_b = mt_w, mt_b
        prev_vt_w, prev_vt_b = vt_w, vt_b

        curr_error = error(w, b)
        errors.append(curr_error)
        epochs.append(epoch)

        if epoch == 999:
            y_pred = [f(x, w, b) for x in X]
            plt.plot(X, Y, 'ro', label='True Y')
            plt.plot(X, Y, 'b-', label='Predicted Y')
            plt.title(f'Epoch {epoch}')
            plt.legend()
            plt.show()

    plt.plot(epochs, errors)
    plt.title('Error over epochs_adamax')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    do_adamax()