# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
idvc = 4
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, idvc].values

# Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Column location(s) of categorical variables
categorical_features_column = 3

# Turns the strings into numbers so OneHotEncoder can separate them out since OneHotEncoder only works with integers
label_encoder_X = LabelEncoder()
X[:, categorical_features_column] = label_encoder_X.fit_transform(X[:, categorical_features_column])

# OneHotEncoder will treat all the columns as categorical variables by default, so we pass an array with the indices into the categorical_features paramter to encode only that column
one_hot_encoder = OneHotEncoder(categorical_features=[categorical_features_column])

# Returns three categorical columns - one for each country - as a numpy array (matrix)
X = one_hot_encoder.fit_transform(X).toarray()

# Avoid the Dummy Variable Trap
# The library takes care of this already, so this line is a reminder
X = X[:, 1:]

# Print variables rounded to 2 decimal places
# print np.around(X, decimals=2)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
# Set Test set size as fraction of entire dataframe
# In Python 2.x, it is necessary to include a decimal place in order to induce a float; In Python 3.x, any division induces a float
test_size = 0.2
# Assign Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

# Fit simple linear regression to the training set
from sklearn.linear_model import LinearRegression

# Instantiate linear regression
regressor = LinearRegression()

# Fit the regression to our training set
regressor.fit(X_train, y_train)

# print X_train
# print y_train

# The predicted profits
y_pred = regressor.predict(X_test)

# print "------ Predicted Profits ------"
# print y_pred
# print "------ Actual Profits ------"
# print y_test

# Build the optimal model using Backward Elimination
# Statsmodels does not take into account the b(0) constant in an MLR model, so we have to add a column in our dataset
import statsmodels.formula.api as sm

# Add column of constants b(0) to the matrix
array_to_add_to = np.ones((50, 1)).astype(int)
values_to_add_to_array = X

X = np.append(arr=array_to_add_to, values=values_to_add_to_array, axis=1)

# print X

# Create the optimal matrix of features
array_of_independent_variables = [0, 1, 2, 3, 4, 5]

X_opt = X[:, array_of_independent_variables]

endogenous_response_variable = y # the dependent variable
exogenous_variables = X_opt # the equation for the MVLR line

# Create new Ordinary Least Squares regression model
regressor_OLS = sm.OLS(endog=endogenous_response_variable, exog=exogenous_variables).fit()

# print "----- First Attempt ------"
# print regressor_OLS.summary()

# Remove the variable with the highest P-value to improve the creation of the optimal matrix of features
array_of_independent_variables = [0, 3]

X_opt = X[:, array_of_independent_variables]

endogenous_response_variable = y # the dependent variable
exogenous_variables = X_opt # the equation for the MVLR line

# Create new Ordinary Least Squares regression model
regressor_OLS = sm.OLS(endog=endogenous_response_variable, exog=exogenous_variables).fit()

print regressor_OLS.summary()