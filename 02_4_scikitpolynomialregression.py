# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('./datasets/Position_Salaries.csv') 
X = dataset[['Level']].to_numpy()
y = dataset['Salary'].to_numpy()

# Training the Polynomial Regression model on the whole dataset
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Predicting a new result 
print(lin_reg.predict(poly_feat.fit_transform([[6.5]])))

# Visualising the Polynomial Regression results
plt.scatter(X, y, color='red')
X_grid = np.arange(X.min(), X.max(), 0.1).reshape(-1, 1)
plt.plot(X_grid, lin_reg.predict(poly_feat.fit_transform(X_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()