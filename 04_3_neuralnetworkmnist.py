import numpy as np
import pandas as pd

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / expZ.sum(axis=1, keepdims=True)

def create_batches(X, y, batch_size):
    batches = []
    n = len(X)
    for start_index in range(0, n, batch_size):
        end_index = min(start_index + batch_size, n)            
        X_batch = X[start_index:end_index, :]
        y_batch = y[start_index:end_index]
        batches.append((X_batch, y_batch))
    return batches

class NeuralNetwork:

    def __init__(self, no_inputs, no_hidden, no_outputs):
        self.W1 = np.random.rand(no_inputs, no_hidden) * 0.1
        self.b1 = np.random.rand(no_hidden) * 0.1
        self.W2 = np.random.rand(no_hidden, no_outputs) * 0.1
        self.b2 = np.random.rand(no_outputs) * 0.1

    def feedforward(self, inputs):
        self.Z1 = inputs @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)

    def backpropagate(self, X, y):
        n = X.shape[0]
        A2_errors = self.A2.copy()
        A2_errors[range(n), y] -= 1
        self.W2_gradients = (self.A1.T @ A2_errors) / n   
        self.b2_gradients = A2_errors.sum(axis=0) / n
        A1_errors = A2_errors @ self.W2.T
        A1_errors *= (self.Z1 > 0)
        self.W1_gradients = (X.T @ A1_errors) / n
        self.b1_gradients = A1_errors.sum(axis=0) / n     

    def update_parameters(self, learning_rate):
        self.W1 -= learning_rate * self.W1_gradients
        self.b1 -= learning_rate * self.b1_gradients
        self.W2 -= learning_rate * self.W2_gradients
        self.b2 -= learning_rate * self.b2_gradients

    def train(self, X, y, epochs, learning_rate, batch_size):
        for _ in range(epochs):
            batches = create_batches(X, y, batch_size)
            for X_batch, y_batch in batches:
                self.feedforward(X_batch)
                self.backpropagate(X_batch, y_batch)
                self.update_parameters(learning_rate)
            print(".", end="", flush=True)

dataset = pd.read_csv("./datasets/mnist.csv").sample(frac=1)

X = dataset.iloc[:, 1:].to_numpy()
y = dataset.iloc[:, 0].to_numpy()

X = X / 255.0

train_sample_size = 3500
test_sample_size = 1000

X_train = X[:train_sample_size]
y_train = y[:train_sample_size]
X_test = X[train_sample_size:train_sample_size + test_sample_size]
y_test = y[train_sample_size:train_sample_size + test_sample_size]

nn = NeuralNetwork(X_train.shape[1], 256, 10)
nn.train(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)

nn.feedforward(X_test)
y_prediction = np.argmax(nn.A2, axis=1)
correct_count = np.sum(y_prediction == y_test)

print(f"Accuracy: {(100 * correct_count / len(X_test)):.1f}%")
