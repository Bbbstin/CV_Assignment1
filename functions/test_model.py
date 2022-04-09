from torchvision import datasets
import numpy as np
from torch import load
import matplotlib.pyplot as plt
from MNIST_loader import *
# If you want to test the model without bias, do not make any changes.
# If you want to test the mdoel with bias, annotate line 9, 31, and de-annotate line 10, 32
# Do NOT de-annotate both of them!!
from NNs import *
# from NNs_bias import *



# If to show the plot, since when need to show the plot, the process will be much slower.
# Therefore, we turn it off here. If need to check, you can just change it to True.
show_plot = False

# Load the MNIST datasets. We split 5,000 samples from the training set as the validation set.
# X_train_flatten.shape = (55000, 784)
# X_val_flatten.shape = (5000, 784)
# X_test_flatten.shape = (10000, 784)
data = MNIST_loader()
X_train_flatten = data[0]
X_val_flatten = data[1]
X_test_flatten = data[2]
Y_train_onehot = data[3]
Y_val_onehot = data[4]
Y_test_onehot = data[5]

# Load the parameters and hyperparameters, which perform the best in the VALIDATION set.
parameters = load("../params")
# parameters = load("../params_bias")
W = parameters[0]
activation = parameters[1]
step_size = parameters[2]
hidden_size = parameters[3]
reg = parameters[4]
epochs = 240000

# Test the model in the test set, and show its accuracy in the test set.
nn_best = Neural_Network([784, hidden_size, 10], epochs, activation=activation, step_size=step_size, reg=reg)
nn_best.params = W
y_hat = nn_best.predict(X_test_flatten)
accuracy = nn_best.accuracy(X_test_flatten, Y_test_onehot)
print("The accuracy of the model in the test set is: {0:.2f}%.".format(accuracy * 100))

# Show the accuracy and loss plots here.
if show_plot == True:
    print("\nRetrain and show the accuracy and loss plot.")
    nn_best_retrain = Neural_Network([784, hidden_size, 10], epochs, activation=activation, step_size=step_size, reg=reg)
    acc = nn_best_retrain.train(X_train_flatten, Y_train_onehot, X_test_flatten, Y_test_onehot, need_plot=True)

    nn_best_retrain.show_accuracy_plot()
    plt.pause(6)
    plt.close()
    nn_best_retrain.show_loss_plot()
    plt.pause(6)
