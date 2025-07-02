import numpy as np
import pandas as pd

# Load the csv file and convert to a numpy array
data = pd.read_csv('./datasets/Salary_Data.csv').to_numpy()

# Work out the number of rows
row_count = data.shape[0]

# The independent variable matrix is up to, but not including, the last column
X = data[:, :-1] 
# Create an extra column of 1s for the intercepts…
intercept_column = np.ones((row_count, 1))
# …stick these onto the end of the matrix 
X = np.hstack([X, intercept_column])

# The dependent variable vector is just the last column
y = data[:, -1]
# Reshape y to a column vector so matrix multiplication works as expected
y = y.reshape(row_count, 1)

# We'll use an 80% train, 20% test dataset split:
split_index = int(0.8 * row_count)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Now we'll carry out the 'Least Squares' method…
XᵀX = X_train.T @ X_train
Xᵀy = X_train.T @ y_train
weights = np.linalg.inv(XᵀX) @ Xᵀy

# …to give us our predicted weights vector
y_predicted = X_test @ weights

# Time to see how good our predictions are!
for i in range(len(y_test)):
    error = y_predicted[i][0] - y_test[i][0]
    percentage_error = 100 * error / y_test[i][0]
    print(f"{y_test[i][0]:.0f} -> {y_predicted[i][0]:.2f}\tError: {error:.2f} ({percentage_error:.1f}%)")

# Output the gradient(s) and intercept, i.e. the values for y = mx + c 
coefficients = weights[:-1, 0]
print(f"\nCoefficients (m1, m2, ...): {coefficients}")
intercept = weights[-1, 0]           
print(f"Intercept (c): {intercept}")

# We'll calculate the average error (root mean square error)
# which represents the effective ± accuracy of the model. 
rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
print(f"\nRoot Mean Squared Error on Test Set: {rmse}\n")
