# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('./datasets/Salary_Data.csv')
X = dataset[['YearsExperience']].to_numpy()
y = dataset['Salary'].to_numpy()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3333, random_state=0)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Output the gradient(s) and intercept
coefficients = regressor.coef_
print(f"\nCoefficients (m1, m2, ...): {coefficients}")
intercept = regressor.intercept_
print(f"Intercept (c): {intercept}")

# Predicting the Test set results, and calculate RMS error
y_pred = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRoot Mean Squared Error on Test Set: {rmse}")

# Visualising the results
plt.scatter(X_train, y_train, color='green')
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Train in green, test in red)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
