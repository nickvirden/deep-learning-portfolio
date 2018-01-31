# Random Forest Regression

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fit Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

# Create instance of the Random Forest Regressor model
regressor = RandomForestRegressor(n_estimators=300, random_state=0)

# Fit regressor to dataset
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5)
print y_pred

# Visualising the Random Forest Regression results (higher resolution)
# X_grid = np.arange(min(X), max(X), 0.01)
# X_grid = X_grid.reshape((len(X_grid), 1))
# plt.scatter(X, y, color = 'red')
# plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
# plt.title('Truth or Bluff (Random Forest Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()