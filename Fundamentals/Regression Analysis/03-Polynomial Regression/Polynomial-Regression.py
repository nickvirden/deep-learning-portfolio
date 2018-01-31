# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
idvc = 2
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, idvc].values

# Fit simple linear regression to the dataset
from sklearn.linear_model import LinearRegression

# Instantiate linear regression
lin_reg = LinearRegression()

# Fit the line to the data
lin_reg.fit(X, y)

# Fit polynomial linear regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

# Instantiate polynomial linear regression
poly_reg = PolynomialFeatures(degree=4)

# Transform the independent variable matrix into a matrix of it's polynomial terms
X_poly = poly_reg.fit_transform(X)

# Create a new instance of the Linear Regression
lin_reg_2 = LinearRegression()

# Fit the linear regression to the X_poly matrix
lin_reg_2.fit(X_poly, y) 

# Visualize the Linear Regression results
# plt.scatter(X, y, color='red')
# plt.plot(X, lin_reg.predict(X), color='blue')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary ($)')
# plt.show()

# Visualize the Polynomial Regression results
# Create a continous curve
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape((len(X_grid), 1))
# Plot the results
# plt.scatter(X, y, color='red')
# plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary ($)')
# plt.show()

# Predict a new result with Linear Regression
print "------ Linear Regression Prediction ------"
print lin_reg.predict(6.5)

# Predict a new result with Polynomial Regression
print "------ Polynomial Regression Prediction ------"
print lin_reg_2.predict(poly_reg.fit_transform(6.5))