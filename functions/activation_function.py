# This file contains several activation functions and a helper function (y2y_onehot).
import numpy as np

def y2y_onehot(y):
    '''
    Description: 
        This function is used to convert labels to their one-hot version.
    Input:
        y: Labels of the samples, shape (n, )
    Output: 
        y_onehot: Labels' one-hot version.
    '''
    labels = np.max(y) + 1
    y_onehot = np.zeros((y.shape[0], labels))
    for index, i in enumerate(y):
        y_onehot[index, i] = 1
    return y_onehot


def relu(x):
    # y = max(0, x)
    return np.maximum(0, x)

def relu_backward(x):
    # dy/dx = 1, if x > 0
    # dy/dx = 0, if x <= 0
    dx = np.zeros(x.shape)
    dx[x > 0] = 1
    return dx

def sigmoid(x):
    # y = 1 / (1 + e^{-x})
    return 1 / (1 + np.exp(-x))

def sigmoid_backward(x):
    # dy/dx = e^{-x} / (e^{-x} + 1)^2
    dx = np.exp(-x) / (np.exp(-x)+1)**2
    return dx

def softmax(x):
    # y_i = e^{x_i} / sum(e^{x})
    # To avoid deviding by 0, we add a small perturbation here.
    exps = np.exp(x) + 1e-6
    if (len(x.shape) == 1):
        return exps / np.sum(exps, axis=0)
    else:
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

def softmax_backward(x):
    # dy_i/dx_j = y_i - y_i^2, if i = j
    #           = - y_i * y_j, if i != j
    exps = np.exp(x) + 1e-6
    if (len(x.shape) == 1):
        dx = exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return dx
    else:
        dx = exps / np.sum(exps, axis=1).reshape(-1, 1) * (1 - exps / np.sum(exps, axis=0).reshape(-1, 1))
        return dx