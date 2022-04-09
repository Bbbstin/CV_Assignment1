import numpy as np
from torch import save
from activation_function import *
from MNIST_loader import *
# If you want to train the model without bias, do not make any changes.
# If you want to train the mdoel with bias, annotate line 8, 67, and de-annotate line 9, 68.
# Do NOT de-annotate both of them!!
from NNs import *
# from NNs_bias import *

silent = True
# Hyperparamters to test.
# reg is the intensity of the regulatization (lambda).
step_size = [1e-4, 5e-4, 1e-3, 1e-2, 5e-2, 1e-1]
reg = [1e-8, 1e-7, 1e-6, 1e-5]
hidden_size = [150, 180, 200, 225, 250, 275, 300]
activation = ["relu", "sigmoid"]
epochs = 240000

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

# Store the best accuracy in the VALIDATION set and its hyperparameters.
accuracy_best = 0
aa_best, ss_best, rr_best, hh_best = None, None, None, None

# Training process
params = {}
count = 0
n_models = len(step_size) * len(reg) * len(hidden_size) * len(activation)
for aa in activation:
    for ss in step_size:
        for rr in reg:
            for hh in hidden_size:
                count = count + 1
                print("\nModel {0}/{1}".format(count, n_models))
                print("Activation function: {0}, step size: {1}, hidden size: {2}, lambda: {3}".format(aa, ss, hh, rr))
                nn = Neural_Network([784, hh, 10], epochs, activation=aa, step_size=ss, reg=rr, silent=silent)
                acc = nn.train(X_train_flatten, Y_train_onehot, X_val_flatten, Y_val_onehot)
                if acc > accuracy_best:
                    accuracy_best = acc
                    params = nn.params
                    aa_best = aa
                    ss_best = ss
                    hh_best = hh
                    rr_best = rr

# Print the best accuracy in the VALIDATION set and its hyperparameters.
print("\nBest accuracy in the validation set: {0:.2f}%".format(accuracy_best * 100))
print("The model used: Activation function: {0}, step size: {1}, hidden size: {2}, lambda: {3}".format(aa_best, ss_best, hh_best, rr_best))
# Save the parameters and hyperparameters.
# We store it as a list:
# 1st element: parameters of the "best" model.
# 2nd element: the model's activation function.
# 3rd element: the model's step size.
# 4th element: the hidden layer's size.
# 5th element: the regularization intensity of l2 (lambda).
save([params, aa_best, ss_best, hh_best, rr_best], '../params')
# save([params, aa_best, ss_best, hh_best, rr_best], '../params_bias')
