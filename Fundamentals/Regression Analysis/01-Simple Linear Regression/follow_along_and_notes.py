# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
idvc = 1
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, idvc].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# Set Test set size as fraction of entire dataframe
# In Python 2.x, it is necessary to include a decimal place in order to induce a float; In Python 3.x, any division induces a float
test_size = 1.0/3
# Assign Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# Fit simple linear regression to the training set
from sklearn.linear_model import LinearRegression

# Instantiate linear regression
regressor = LinearRegression()

# Fit the regression to our training set
regressor.fit(X_train, y_train)

print X_train
print y_train

# Predict the Test set results
# y_pred is the vector of predictions
# Here, we're making predictions on X_test based on the data in X_train and y_train
y_pred = regressor.predict(X_test)

# y_test is the real salary
# y_pred is the predicted salary

# Visualize the Training set results
# Plot the training set to which we fit our SLR
plt.scatter(X_train, y_train, color='red')

# Plot the regression line on top of the scatter plot in a different color
plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.show()