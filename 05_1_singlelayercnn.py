import numpy as np
import pandas as pd

import sys
import time
import itertools

spinner_chars = itertools.cycle(['/', '-', '\\', '|'])

def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    column_sums = np.sum(expZ, axis=0)
    return expZ / column_sums

def softmax_derivative(Z):
    s = Z.reshape(-1, 1)
    return np.diagflat(s) - s @ s.T


def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

class ConvolutionalLayer:
    def __init__(self, num_filters, filter_size, num_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size, num_channels) / (filter_size * filter_size)
                        
    def feedforward(self, input_image):        
        self.last_input = input_image
        h, w, _ = input_image.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        self.regions = []
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                region = input_image[i:(i + self.filter_size), j:(j + self.filter_size)]
                self.regions.append((region, i, j))

        for region, i, j in self.regions:
            output[i, j] = np.sum(region * self.filters, axis=(1, 2, 3))

        self.output = relu(output)
        return self.output
    
    def backpropagate(self, output_gradients, learning_rate):

        output_gradients = output_gradients * relu_derivative(self.output)
        filter_gradients = np.zeros(self.filters.shape)
        input_gradients = np.zeros(self.last_input.shape)

        for region, i, j in self.regions:
            for f in range(self.num_filters):
                filter_gradients[f] += output_gradients[i, j, f] * region
                input_gradients[i:(i + self.filter_size), j:(j + self.filter_size)] += output_gradients[i, j, f] * self.filters[f]    

        self.filters -= learning_rate * filter_gradients

        return input_gradients

class MaxPoolingLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size
            
    def feedforward(self, input_image):
        self.last_input = input_image
        h, w, num_filters = input_image.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))
        
        self.regions = []
        for i in range(h // self.pool_size):
            for j in range(w // self.pool_size):
                region = input_image[(i * self.pool_size):(i * self.pool_size + self.pool_size),
                               (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                self.regions.append((region, i, j))
        
        for region, i, j in self.regions:
            output[i, j] = np.amax(region, axis=(0, 1))
            
        return output
    
    def backpropagate(self, output_gradients):
        input_gradients = np.zeros(self.last_input.shape)
        
        for region, i, j in self.regions:
            h, w, f = region.shape
            amax = np.amax(region, axis=(0, 1))
            
            for x in range(h):
                for y in range(w):
                    for z in range(f):
                        if region[x, y, z] == amax[z]:
                            input_gradients[i * self.pool_size + x, j * self.pool_size + y, z] = output_gradients[i, j, z]
                            
        return input_gradients

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros(output_size)
        self.activation_function = activation_function
        
    def feedforward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        self.last_totals = input @ self.weights + self.biases
        
        if self.activation_function == "softmax":
            self.last_output = softmax(self.last_totals)
        elif self.activation_function == "relu":
            self.last_output = relu(self.last_totals)
        return self.last_output
        
    def backpropagate(self, output_gradients, learning_rate):
        if self.activation_function == "softmax":
            loss_to_totals_gradients = output_gradients @ softmax_derivative(self.last_output)
        elif self.activation_function == "relu":
            loss_to_totals_gradients = output_gradients * relu_derivative(self.last_totals)
        
        input_to_weights_gradients = self.last_input[:, np.newaxis]
        weight_gradients = input_to_weights_gradients @ loss_to_totals_gradients[np.newaxis, :]
        bias_gradients = loss_to_totals_gradients

        input_gradients = loss_to_totals_gradients @ self.weights.T

        self.weights -= learning_rate * weight_gradients
        self.biases -= learning_rate * bias_gradients

        return input_gradients.reshape(self.last_input_shape)

class ConvolutionalNN:
    def __init__(self, num_filters, filter_size, num_classes):
        self.convolutional_layer = ConvolutionalLayer(num_filters, filter_size, 1)  
        self.max_pooling_layer = MaxPoolingLayer(2)  
        self.fully_connected_layer = FullyConnectedLayer(13 * 13 * num_filters, num_classes, "softmax")
        self.num_classes = num_classes
        
    def feedforward(self, input):
        output = self.convolutional_layer.feedforward(input)
        output = self.max_pooling_layer.feedforward(output)
        output = self.fully_connected_layer.feedforward(output)
        return output
    
    def train(self, X, y, epochs, learning_rate):

        start_time = time.time()
        
        for epoch in range(epochs):
            l = len(X)
            for i in range(l):
                
                output = self.feedforward(X[i])
                gradient = np.zeros(self.num_classes)
                epsilon = 1e-12
                correct_index = y[i]
                gradient[correct_index] = -1 / (output[correct_index] + epsilon)

                gradient = self.fully_connected_layer.backpropagate(gradient, learning_rate)
                gradient = self.max_pooling_layer.backpropagate(gradient)
                self.convolutional_layer.backpropagate(gradient, learning_rate)
                
                if i % 25 == 0 and (i > 0 or epoch > 0):
                        
                    elapsed_time = time.time() - start_time
                    
                    number_done = i + epoch * l
                    total = l*epochs
                    
                    per_item = elapsed_time / number_done
                    total_eta = total * per_item                
                    eta = total_eta - elapsed_time
                
                    progress = f"{next(spinner_chars)} {int(100 * number_done / total)}% ... ETA: {int(eta/60)} minutes {int(eta) - 60*int(eta/60)} seconds" + " " * 40
                    sys.stdout.write(progress[:40]) 
                    sys.stdout.flush()  
                    sys.stdout.write('\b' * 40) 

            print(f"Epoch {epoch + 1} done at {int(elapsed_time/60)} minutes {int(elapsed_time) - 60*int(elapsed_time/60)} seconds")


num_classes = 10
filter_size = 3

dataset = pd.read_csv('./datasets/mnist.csv') 
X = dataset.iloc[:, 1:].to_numpy()
y = dataset.iloc[:, 0].to_numpy()
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

train_sample_size = 500
test_sample_size = 1000

X_train = X[:train_sample_size]
y_train = y[:train_sample_size]
X_test = X[train_sample_size:train_sample_size+test_sample_size]
y_test = y[train_sample_size:train_sample_size+test_sample_size]

num_filters = 5
epochs = 10
learning_rate = 0.01

cnn = ConvolutionalNN(num_filters, filter_size, num_classes)
cnn.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)

correct_predictions = 0
for i in range(len(X_test)):
    output = cnn.feedforward(X_test[i])
    if np.argmax(output) == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(X_test)
print(f'Test accuracy: {accuracy * 100:.2f}%')
