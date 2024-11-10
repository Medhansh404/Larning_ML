from Activation import sigmoid
from Errors import mse
from utils import display
import numpy as np


X = [0.5, 1.0,  2.5]
Y = [0.2, 0.84, 0.9]


def do_gradient_descent():
    w, b, eta, max_epochs = -2, -2, 1.0, 1000
    epochs, errors = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += sigmoid.grad_w(x, w, b, y)
            db += sigmoid.grad_b(x, w, b, y)

        w = w - eta * dw
        b = b - eta * db

        current_error = mse.error(w, b, X, Y)
        errors.append(current_error)
        epochs.append(i)
        if i % 100 == 0:
            Y_pred = [sigmoid.f(x, w, b) for x in X]
            display.display_neuron(w, b, i, X, Y, Y_pred)

    display.display_error(epochs, errors)


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
            dw += sigmoid.grad_w(w, b, x, y)
            db += sigmoid.grad_b(w, b, x, y)

        v_w = beta * v_w + (1 - beta) * dw ** 2
        v_b = beta * v_b + (1 - beta) * db ** 2

        delta_w = dw * np.sqrt(u_w + eps) / (np.sqrt(v_w + eps))
        delta_b = db * np.sqrt(u_b + eps) / (np.sqrt(v_b + eps))
        u_w = beta * u_w + (1 - beta) * delta_w ** 2
        u_b = beta * u_b + (1 - beta) * delta_b ** 2

        w = w - delta_w
        b = b - delta_b

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)


def do_adagrad():
    max_epochs = 1000
    w, b, eta, eps, ub, uw = -2, -2, 1.0, 0.00000001, 0.0, 0.0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += sigmoid.grad_w(x, y, w, b)
            db += sigmoid.grad_b(x, y, w, b)
        uw = uw + dw ** 2
        ub = ub + db ** 2
        w = w - eta * dw / (np.sqrt(uw) + eps)
        b = b - eta * db / (np.sqrt(ub) + eps)

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)


def do_adam():
    max_epochs, w, b, eta, e = 1000, -2.0, -2.0, 0.1, .0000001
    beta1, beta2 = 0.5, 0.5
    errors, epochs = [], []
    prev_mt_w, prev_mt_b, prev_vt_w, prev_vt_b = 0, 0, 0, 0
    for epoch in range(max_epochs):
        dw, db = 0, 0
        for (x, y) in zip(X, Y):
            dw += sigmoid.grad_w(x, y, w, b)
            db += sigmoid.grad_b(x, y, w, b)

        mt_w = beta1 * prev_mt_w + (1 - beta1) * dw
        mt_b = beta1 * prev_mt_b + (1-beta1) * db
        vt_w = beta2 * prev_vt_w + (1 - beta2) * (dw ** 2)
        vt_b = beta2 * prev_vt_b + (1 - beta2) * (db ** 2)

        vt_w /= (1 - beta2)
        vt_b /= (1 - beta2)
        mt_w /= 1 - beta1
        mt_b /= 1 - beta1

        w = w - eta * mt_w/(np.sqrt(vt_w)+e)
        b = b - eta * mt_b/(np.sqrt(vt_b)+e)

        prev_mt_w, prev_mt_b = mt_w, mt_b
        prev_vt_w, prev_vt_b = vt_w, vt_b

        curr_error = mse.error(w, b)
        errors.append(curr_error)
        epochs.append(epoch)


def do_adamax():
    max_epochs, w, b, eta, e = 1000, -2.0, -2.0, 0.1, .00000001
    beta1, beta2 = 0.1, 0.1
    errors, epochs = [], []
    prev_mt_w, prev_mt_b, prev_vt_w, prev_vt_b = 0, 0, 0, 0
    for epoch in range(max_epochs):
        dw, db = 0, 0
        for (x, y) in zip(X, Y):
            dw += sigmoid.grad_w(x, y, w, b)
            db += sigmoid.grad_b(x, y, w, b)

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

        curr_error = mse.error(w, b)
        errors.append(curr_error)
        epochs.append(epoch)


def do_maxprop():
    max_epochs = 1000
    w, b, eta, beta, eps = -2, -2, 0.1, 0.5, 0.000000001
    prev_ub, prev_uw = 0, 0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += sigmoid.grad_w(x, y, w, b)
            db += sigmoid.grad_b(x, y, w, b)
        uw = max(beta * prev_uw, abs(dw))
        ub = max(beta * prev_ub, abs(db))

        w = w - eta * dw / (uw + eps)
        b = b - eta * db / (ub + eps)

        prev_ub = ub
        prev_uw = uw

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)

    max_epochs = 1000
    errors, epochs = [], []

    w, b, eta = -2, -2, 1.0
    prev_uw, prev_ub, beta = 0, 0, 0.9

    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += sigmoid.grad_w(w, b, x, y)
            db += sigmoid.grad_b(w, b, x, y)

        uw = beta * prev_uw + eta * dw
        ub = beta * prev_ub + eta * db
        w = w - uw
        b = b - ub
        prev_uw = uw
        prev_ub = ub

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)


def do_mini_batch_gd():
    batch_size, eta, max_epochs, w, b = 3, 0.1, 1000, -2.0, -2.0
    epochs, errors = [], []
    for epoch in range(max_epochs):
        count = 0
        db, dw = 0, 0
        for (x,y) in zip(X,Y):
            count += 1
            dw += sigmoid.grad_w(x, y, w, b)
            db += sigmoid.grad_b(x, y, w, b)
            if count == batch_size:
                w = w - eta*dw
                b = b - eta*db
                db, dw = 0, 0

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(epoch)


def do_msgd():
    max_epochs = 1000
    w, b, beta, eta = -2, -2, 0.9, 1.0
    prev_wu, prev_bu = 0, 0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw = sigmoid.grad_w(x, y, w, b)
            db = sigmoid.grad_b(x, y, w, b)
            uw = prev_wu * beta + eta * dw
            ub = prev_bu * beta + eta * db
            w = w - uw
            b = b - ub
            prev_wu = uw
            prev_bu = ub

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)

def do_ngd():
    max_epoch = 1000
    errors, epochs = [], []

    w, b, eta = -2, -2, 1.0
    prev_vw, prev_vb, beta = 0, 0, 0.9

    for i in range(max_epoch):
        dw, db = 0, 0
        v_w = beta * prev_vw
        v_b = beta * prev_vb
        for x, y in zip(X, Y):
            dw += sigmoid.grad_w(w - v_w, b - v_b, x, y)
            db += sigmoid.grad_b(w - v_w, b - v_b, x, y)
        vw = beta * prev_vw + eta * dw
        vb = beta * prev_vb + eta * db
        w = w - vw
        b = b - vb
        prev_vw = vw
        prev_vb = vb

        current_error = error(w, b)
        errors.append(current_error)
        epochs.append(i)

def do_nsgd():
    max_epochs = 1000
    w, b, beta, eta = -2, -2, 0.9, 1.0
    prev_wu, prev_bu = 0, 0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        vw, vb = beta * prev_wu, beta * prev_bu
        for x, y in zip(X, Y):
            dw = sigmoid.grad_w(x, y, w - vw, b - vb)
            db = sigmoid.grad_b(x, y, w - vw, b - vb)
            uw = beta * prev_wu + eta * dw
            ub = beta * prev_bu + eta * db
            w = w - uw
            b = b - ub
            prev_wu = uw
            prev_bu = ub

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)

def do_rmsprop():
    max_epochs = 1000
    w, b, eta, beta, eps = -2, -2, 0.1, 0.5, 0.000000001
    ub, uw = 0, 0
    errors, epochs = [], []
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += sigmoid.grad_w(x, y, w, b)
            db += sigmoid.grad_b(x, y, w, b)
        uw = beta * uw + (1 - beta) * dw ** 2
        ub = beta * ub + (1 - beta) * db ** 2
        w = w - eta * dw/ (np.sqrt(uw) + eps)
        b = b - eta * db/ (np.sqrt(ub) + eps)

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)


def do_sgd():
    w, b, n, max_epoch = -2, -2, 1.0, 1000
    errors, epochs = [], []
    for i in range(max_epoch):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += sigmoid.grad_w(x, w, b, y)
            db += sigmoid.grad_b(x, w, b, y)
            w = w - n * dw
            b = b - n * db

        current_error = mse.error(w, b)
        errors.append(current_error)
        epochs.append(i)


if __name__ == "__main__":
    do_gradient_descent()