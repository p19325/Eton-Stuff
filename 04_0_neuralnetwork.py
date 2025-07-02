import numpy as np
import pandas as pd
    
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)

def feedforward(X):
    global Z1, A1, Z2, A2
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

def backpropagate(X, y):
    global W1_gradients, b1_gradients, W2_gradients, b2_gradients
    n = X.shape[0]

    A2_errors = A2.copy()
    A2_errors[range(n), y] -= 1

    W2_gradients = A1.T @ A2_errors / n
    b2_gradients = A2_errors.sum(axis=0) / n

    A1_errors = A2_errors @ W2.T
    A1_errors *= (Z1 > 0)
    
    W1_gradients = X.T @ A1_errors / n
    b1_gradients = A1_errors.sum(axis=0) / n

def update_parameters(learning_rate):
    global W1, b1, W2, b2
    W1 -= learning_rate * W1_gradients
    b1 -= learning_rate * b1_gradients
    W2 -= learning_rate * W2_gradients
    b2 -= learning_rate * b2_gradients

def train(X, y, epochs, learning_rate):
    for _ in range(epochs):
        feedforward(X)
        backpropagate(X, y)
        update_parameters(learning_rate)

# Load and shuffle the dataset
dataset = pd.read_csv("./datasets/iris.csv").sample(frac=1)

# Extract input features and convert class labels to integers
X = dataset[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
y, _ = dataset["class"].factorize()

# Split into training and testing data
split_index = int(len(X) * 0.75)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Initialize parameters
W1 = np.random.rand(4, 10) * 0.1
b1 = np.random.rand(10) * 0.1
W2 = np.random.rand(10, 3) * 0.1
b2 = np.random.rand(3) * 0.1

# Train the model
train(X_train, y_train, epochs=1000, learning_rate=0.01)

# Evaluate
feedforward(X_test)
y_prediction = np.argmax(A2, axis=1)
correct_count = np.sum(y_prediction == y_test)
accuracy = 100 * correct_count / len(y_test)
print(f"Accuracy: {accuracy:.1f}%")
