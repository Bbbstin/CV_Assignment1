from torchvision import datasets
import numpy as np
from activation_function import *

# Load MNIST, split 5,000 samples from training set as validation set.
def MNIST_loader():
    '''
    output:
        X_train_flatten: Flattened training set, shape (55000, 784).
        X_val_flatten: Flattened validation set, which is splitted from the training set, shape (5000, 784).
        X_test_flatten: Flattened test set, shape (10000, 784).
        Y_train_onehot: Labels of the training set (One-hot), shape (55000, 10).
        Y_val_onehot: Labels of the validation set (One-hot), shape (5000, 10).
        Y_test_onehot: Labels of the test set (One-hot), shape (10000, 10).
    '''
    train_set = datasets.MNIST('./data', train=True, download=True)
    test_set = datasets.MNIST('./data', train=False, download=True)

    train_set_array = train_set.data.numpy()
    test_set_array = test_set.data.numpy()
    train_set_array_targets = train_set.targets.numpy()
    test_set_array_targets = test_set.targets.numpy()

    X_train = ((train_set_array / 255) - 0.1307) / 0.3081
    Y_train = train_set_array_targets
    X_test = ((test_set_array / 255) - 0.1307) / 0.3081
    Y_test = test_set_array_targets

    # Split 5,000 examples from X_train as validation set.
    index = [i for i in range(X_train.shape[0])]
    np.random.shuffle(index)
    X_val = X_train[index[0:5000], :, :]
    Y_val = Y_train[index[0:5000]]
    X_train = X_train[index[5000:60000], :, :]
    Y_train = Y_train[index[5000:60000]]

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]
    dim_per = train_set_array.shape[1]
    dim = dim_per * dim_per

    X_train_flatten = X_train.reshape(n_train, dim)
    X_val_flatten = X_val.reshape(n_val, dim)
    X_test_flatten = X_test.reshape(n_test, dim)
    Y_train_onehot = y2y_onehot(Y_train)
    Y_val_onehot = y2y_onehot(Y_val)
    Y_test_onehot = y2y_onehot(Y_test)
    return [X_train_flatten, X_val_flatten, X_test_flatten, Y_train_onehot, Y_val_onehot, Y_test_onehot]