import numpy as np
import pandas as pd

# Activation functions
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / expZ.sum(axis=1, keepdims=True)

class NeuralNetwork:

    def __init__(self, no_inputs, no_hidden, no_outputs):
        # Initialize weights and biases with small random values
        self.W1 = np.random.rand(no_inputs, no_hidden) * 0.1
        self.b1 = np.random.rand(no_hidden) * 0.1
        self.W2 = np.random.rand(no_hidden, no_outputs) * 0.1
        self.b2 = np.random.rand(no_outputs) * 0.1

    def feedforward(self, inputs):
        # Input to hidden layer
        self.Z1 = inputs @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        # Hidden layer to output layer
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)

    def backpropagate(self, X, y):
        n = X.shape[0]  # number of data points

        # Error at output layer
        A2_errors = self.A2.copy()
        A2_errors[range(n), y] -= 1  # Subtract 1 from outputs that are supposed to be 1

        # Gradients for output layer
        self.W2_gradients = (self.A1.T @ A2_errors) / n
        self.b2_gradients = A2_errors.sum(axis=0) / n     

        # Backpropagate error to hidden layer
        A1_errors = A2_errors @ self.W2.T  # Distribute the error backward through the network
        A1_errors *= (self.Z1 > 0)  # Multiply by 1 if input to ReLU was > 0, else multiply by 0

        # Gradients for hidden layer
        self.W1_gradients = (X.T @ A1_errors) / n
        self.b1_gradients = A1_errors.sum(axis=0) / n     

    def update_parameters(self, learning_rate):
        # Update weights and biases using the computed gradients
        self.W1 -= learning_rate * self.W1_gradients
        self.b1 -= learning_rate * self.b1_gradients
        self.W2 -= learning_rate * self.W2_gradients
        self.b2 -= learning_rate * self.b2_gradients

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            self.feedforward(X)
            self.backpropagate(X, y)
            self.update_parameters(learning_rate)

# Load and shuffle the dataset
dataset = pd.read_csv("./datasets/iris.csv").sample(frac=1)

# Extract input features and convert class labels to integers
X = dataset[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
y, _ = dataset["class"].factorize()

# Split into training and testing data
split_index = int(len(X) * 0.75)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Create and train the neural network
nn = NeuralNetwork(4, 10, 3)
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Test the trained model
nn.feedforward(X_test)
y_prediction = np.argmax(nn.A2, axis=1)
correct_count = np.sum(y_prediction == y_test)
accuracy = 100 * correct_count / len(y_test)
print(f"Accuracy: {accuracy:.1f}%")
