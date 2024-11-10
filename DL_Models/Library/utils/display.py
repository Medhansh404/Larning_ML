import matplotlib.pyplot as plt


def display_error(epochs, errors):
    plt.plot(epochs, errors)
    plt.title('Errors over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Errors')
    plt.show()


def display_neuron(w, b, i, X, Y, Y_pred):
    plt.plot(X, Y, 'ro', label='True Y')  # True Y values
    plt.plot(X, Y_pred, 'b-', label='Predicted Y')  # Predicted Y values
    plt.title(f'Epoch {i}')
    plt.legend()
    plt.show()