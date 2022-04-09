import numpy as np
import matplotlib.pyplot as plt
from activation_function import *

# This is the two layers NNs model.
class Neural_Network():
    def __init__(self, sizes, epochs, activation="sigmoid", step_size=0.01, reg=0.0001, silent=False):
        '''
        Description: 
            Init the weights of each layer, and we use Xavier initializaion for better performance.
        Input:
            sizes: A list, the first element is the input size, the second one is hidden size, the last
                    one is the output size. len(sizes) = 3
            epochs: # of iterations to train.
            activation: Activation function, can choose "sigmoid" or "relu", default is "sigmoid".
            step_size: Step size, default value 0.01.
            reg: Intensity of the l2 regularization, default 0.0001.
            silent: If to output the training process, default is False (output the training process).
        '''
        self.epochs = epochs
        self.step_size = step_size
        self.activation = activation
        self.reg = reg
        self.silent = silent

        input_layer = sizes[0]
        hidden_layer = sizes[1]
        output_layer = sizes[2]

        # parameters
        self.params = {
            'W1': np.random.randn(input_layer, hidden_layer) / np.sqrt(hidden_layer),
            'W2': np.random.randn(hidden_layer, output_layer) / np.sqrt(output_layer)
        }


    def forward(self, x):
        '''
        Description:
            Forward compute.
        Input: 
            x: Sample to compute forward pass, shape (n, 784).
        Output:
            memory['A2']: Predicted labels with probability (n, 10)
        '''
        activation = self.activation
        if activation == "sigmoid":
            activation = sigmoid
        elif activation == "relu":
            activation = relu
        else:
            raise Exception('Non-supported activation function')
        params = self.params
        memory = {}
        memory['A0'] = x

        memory['Z1'] = memory['A0'] @ params["W1"]
        memory['A1'] = activation(memory['Z1'])

        memory['Z2'] = memory['A1'] @ params["W2"]
        memory['A2'] = softmax(memory['Z2'])
        self.memory = memory
        return memory['A2']


    def loss(self, output, y_onehot):
        '''
        Description:
            Compute the cross entropy loss and regularization of the model.
        Input:
            output: the predicted labels with probablity of each label.
            y_onehot: the true labels (one-hot).
        Output:
            f: The cross entropy loss and regularization.
        '''
        params = self.params
        reg = self.reg
        n = y_onehot.shape[0]
        y = np.argmax(y_onehot, axis=1)
        f = 0
        for key, value in params.items():
            f = f + reg * np.sum(params[key] ** 2)
        for i in range(n):
            f = f - np.log(output[i, y[i]])
        return f

    def loss_without_reg(self, output, y_onehot):
        '''
        Description:
            Compute the cross entropy loss of the model.
        Input:
            output: the predicted labels with probablity of each label.
            y_onehot: the true labels (one-hot).
        Output:
            f: The cross entropy loss.
        '''
        n = y_onehot.shape[0]
        y = np.argmax(y_onehot, axis=1)
        f = 0
        for i in range(n):
            f = f - np.log(output[i, y[i]])
        return f / n
    

    def backward(self, output, y_train):
        '''
        Description: 
            Compute the gradients of each parameters.
        Input: 
            output: the predicted labels with probablity of each label.
            y_train: the true labels (one-hot).
        Output: 
            dW: the gradient of each parameters.
        '''
        activation = self.activation
        if activation == "sigmoid":
            activation_backward = sigmoid_backward
        elif activation == "relu":
            activation_backward = relu_backward
        else:
            raise Exception('Non-supported activation function')
        params = self.params
        memory = self.memory
        dW = {}

        # Compute dW2.
        # backward = 2 * (output - y_train) / output.shape[0] * softmax_backward(memory['Z2'])
        backward = (output - y_train)
        dW['W2'] = np.outer(memory['A1'], backward) + 2 * self.reg * params['W2']

        # Compute dW1.
        backward = np.dot(params['W2'], backward) * activation_backward(memory['Z1'])
        dW['W1'] = np.outer(memory['A0'], backward) + 2 * self.reg * params['W1']
        return dW


    def learning_rate_decay(self, step):
        '''
        Description: 
            Decay the step_size. (Exponential decay: 0.9^{step/10000}).
        Input:
            step: # epochs have already trained.
        Output:
            Decayed step_size.
        '''
        return self.step_size * 0.9 ** (step / 10000)    


    def predict(self, x):
        '''
        Description: 
            Predict the labels of the samples.
        Input:
            x: Samples.
        Output:
            y_hat: predicted labels.
        '''
        output = self.forward(x)
        y_hat = np.argmax(output, axis=1)
        return y_hat


    def accuracy(self, x_test, y_test):
        '''
        Description:
            Calculate the accuracy of the prediction.
        Input:
            x_test: smaples.
            y_test: true labels (one-hot).
        Output:
            Accuracy.
        '''
        y = np.argmax(y_test, axis=1)
        n = x_test.shape[0]
        correct = 0
        for index, x in enumerate(x_test):
            output = self.forward(x)
            pred = np.argmax(output)
            if pred == y[index]:
                correct = correct + 1
        return correct / n


    def train(self, x_train, y_train_onehot, x_val, y_val_onehot, need_plot=False):
        '''
        Description:
            The training function.
        Input:
            x_train: training set.
            y_train_onehot: training set labels (ont-hot).
            x_val: validation set.
            y_val_onehot: validation set labels (one-hot).
            need_plot: if need to draw the plot later, default is False. (If it is true, the training 
                        process will be much slower.)
        Output:
            Final accuracy in the validation set (test set when we test the accuracy in the final model).
        '''
        n_train = x_train.shape[0]
        epochs = self.epochs
        if need_plot == True:
            # If need plot, need to store the accuracy and loss during the training.
            loss_train_arr = np.zeros(51)
            loss_val_arr = np.zeros(51)
            acc_train_arr = np.zeros(51)
            acc_val_arr = np.zeros(51)
            num = 0

            for iteration in range(epochs):
                rand_i = int(np.random.rand(1) * n_train)
                x_rand = x_train[rand_i, :]

                output = self.forward(x_rand)
                dw = self.backward(output, y_train_onehot[rand_i, :])

                for key, value in dw.items():
                    self.params[key] -= self.learning_rate_decay(iteration) * value
                
                if (iteration % (epochs // 10) == 0) and (self.silent == False):
                    accuracy = self.accuracy(x_val, y_val_onehot)
                    print("Epoch: {0}, Accuracy in the test set: {1:.2f}%".format(iteration+1, accuracy * 100))
                
                if ((iteration % (epochs // 50) == 0)):
                    out_train = self.forward(x_train)
                    out_val = self.forward(x_val)
                    acc_train_arr[num] = self.accuracy(x_train, y_train_onehot)
                    acc_val_arr[num] = self.accuracy(x_val, y_val_onehot)
                    loss_train_arr[num] = self.loss_without_reg(out_train, y_train_onehot)
                    loss_val_arr[num] = self.loss_without_reg(out_val, y_val_onehot)
                    num = num + 1

            out_train = self.forward(x_train)
            out_val = self.forward(x_val)
            acc_train_arr[num] = self.accuracy(x_train, y_train_onehot)
            acc_val_arr[num] = self.accuracy(x_val, y_val_onehot)
            loss_train_arr[num] = self.loss_without_reg(out_train, y_train_onehot)
            loss_val_arr[num] = self.loss_without_reg(out_val, y_val_onehot)
            
            self.acc_train_arr = acc_train_arr
            self.acc_val_arr = acc_val_arr
            self.loss_train_arr = loss_train_arr
            self.loss_val_arr = loss_val_arr
        else:
            for iteration in range(epochs):
                rand_i = int(np.random.rand(1) * n_train)
                x_rand = x_train[rand_i, :]

                output = self.forward(x_rand)
                dw = self.backward(output, y_train_onehot[rand_i, :])

                for key, value in dw.items():
                    self.params[key] -= self.learning_rate_decay(iteration) * value
                
                if (iteration % (epochs // 10) == 0) and (self.silent == False):
                    accuracy = self.accuracy(x_val, y_val_onehot)
                    print("Epoch: {0}, Accuracy in the validation set: {1:.2f}%".format(iteration+1, accuracy * 100))
                    
        accuracy = self.accuracy(x_val, y_val_onehot)
        print("Epoch: {0}, Final accuracy in the validation set: {1:.2f}%".format(iteration+1, accuracy * 100))
        return accuracy


    def show_accuracy_plot(self):
        '''
        Description: 
            Show the accuracy of the training and the test set.
        '''
        epochs = self.epochs
        acc_train = self.acc_train_arr
        acc_val = self.acc_val_arr
        i = np.arange(0, 51, 1) * (epochs / 50)
        plt.plot(i, acc_train, i, acc_val)
        plt.grid()
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["Training set", "Test set"])

    def show_loss_plot(self):
        '''
        Description: 
            Show the loss of the training and the test set.
        '''
        epochs = self.epochs
        loss_train = self.loss_train_arr
        loss_val = self.loss_val_arr
        i = np.arange(0, 51, 1) * (epochs / 50)
        plt.plot(i, loss_train, i, loss_val)
        plt.grid()
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["Training set", "Test set"])