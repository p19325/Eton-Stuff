# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv('./datasets/50_Startups.csv')
X = dataset[['R&D Spend', 'Administration', 'Marketing Spend', 'State']].to_numpy()
y = dataset['Profit'].to_numpy()

# Encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the Multiple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
for i in range(len(y_pred)):
  print(f"{y_pred[i]} predicted vs. {y_test[i]} actual")

# Output the gradient(s) and intercept
coefficients = regressor.coef_
print(f"\nCoefficients (m1, m2, ...): {coefficients}")
intercept = regressor.intercept_
print(f"Intercept (c): {intercept}")

# Calculate RMS error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nRoot Mean Squared Error on Test Set: {rmse}")
