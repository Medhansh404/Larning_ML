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
    fx = f(x, w, b)
    return (fx-y) * (1-x) * x * fx


def grad_b(x, y, w, b):
    fx = f(x, w, b)
    return (fx-y) * (1-x) * fx


def do_mini_batch_gd():
    batch_size, eta, max_epochs, w, b = 3, 0.1, 1000, -2.0, -2.0
    epochs, errors = [], []
    for epoch in range(max_epochs):
        count = 0
        db, dw = 0, 0
        for (x,y) in zip(X,Y):
            count += 1
            dw += grad_w(x, y, w, b)
            db += grad_b(x, y, w, b)
            if count == batch_size:
                w = w - eta*dw
                b = b - eta*db
                db, dw = 0, 0

        current_error = error(w, b)
        errors.append(current_error)
        epochs.append(epoch)

        if epoch == 999:
            Y_pred = [f(x, w, b) for x in X]
            plt.plot(X, Y, 'ro', label='True Y')
            plt.plot(X, Y_pred, 'b-', label='Predicted Y')
            plt.title(f'Epoch {epoch}')
            plt.legend()
            plt.show()

    plt.plot(epochs, errors)
    plt.title('Error over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    do_mini_batch_gd()


