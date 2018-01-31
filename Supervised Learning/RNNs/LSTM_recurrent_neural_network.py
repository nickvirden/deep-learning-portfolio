#################################
### RECURRENT NEURAL NETWORKS ###
#################################

# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the Data Set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# Get all of the values in the second and third columns of the dataframe
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling

# Standardization => (x - mean(x)) / stdev(x)
# Normalization => (x - min(x)) / (max(x) - min(x))

# Normalization is recommended if you have a binary output (i.e. using a sigmoid activation function)

from sklearn.preprocessing import MinMaxScaler

# Since the range > x - min, all of the data will have a value between 0 and 1
sc = MinMaxScaler(feature_range=(0,1))

# Fits sc to the training set AND transforms (scales) the training set
# training_set_scaled is a single-column dataframe with all of the values
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

# Iterate over a the range of days that are tracked by the stock price
for i in range(60, 1258):
	# We want to append the data from the 60 previous days to X_train to create a 60-day moving average. So, since i starts at the 60th day in order for us to have 60 days of data from the past, we take all of the data from previous days as a list and append that within the X_train list
	# X_train will contain all of the past 60 days of stock prices
	X_train.append(training_set_scaled[i-60:i, 0])
	# y_train is a list of all of the t + 1st day of trading that we'll try to predict later
	y_train.append(training_set_scaled[i, 0])

# We need to convert X_train and y_train to numpy arrays so they'll be accepted by the RNN
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# We have to reshape X_train to a 3D numpy array because we have to account for indicators in RNNs
# For Keras, the input shape must be a 3D Tensor (https://keras.io/layers/recurrent/) with batch_size, timesteps, input_dim
# The second argument in np.reshape is the number of rows, columns, and depth of the array
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#####################
### Build the RNN ###
#####################
# This RNN will be a stacked (multi-layer) Long-Short-Term-Memory RNN with Dropout Regularization

# Import Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize the RNN
# We're predicting a continuous value, so we call it a regressor
regressor = Sequential()

# Add the first LSTM layer and some Dropout regularization
# units => number of 'neurons'
#       => we want high dimensionality, so we can include more neurons to account for the complexity of predicting stock prices
# return_sequences => True if you have another new layer, otherwise False (default)
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# rate => percentage of neurons to ignore during the forward propagation and back propagation during each iteration of the training
regressor.add(Dropout('0.2'))

# input_shape is carried over from the first layer, so we don't need to specify it
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout('0.2'))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout('0.2'))

# This is the final layer, so we exclude return_sequences argument
regressor.add(LSTM(units=50))
regressor.add(Dropout('0.2'))

# Output Layer
regressor.add(Dense(units=1))

# Compile RNN
# Keras-recommended optimizers: https://keras.io/optimizers/
# Loss is the Mean-Squared Error (MSE) since we're trying to find the distance between real and predicted values. Also, MSE shows more pronounced differences between small values, such as our normalized values that we have here
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fit RNN to Training Set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# If the loss is too small, then you need to find better ways to optimize your model.

########################
### Make Predictions ###
########################

# Get the real 2017 stock price
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Get the predicted stock price
# This will get an entire dataframe of the open Google stock prices in the training and test sets for Google
# The training set contains the historic stock price for the last 5 years at Google
# The test set contains the first 20 days of stock prices in 2017 for Google
# axis => concatenate along rows or columns?
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)

# The previous stock prices just before the last stock price that we predict (i.e. the last stock price of dataset_total)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

# Transforms the data shape from multiple rows and one column to one row and multiple columns
inputs = inputs.reshape(-1,1)

# We must apply the same scaling to the data so we can compare apples to apples
inputs = sc.transform(inputs)

# Creat a data structure with 60 timesteps and 1 output
X_test = []

# Iterate over the 20 financial days of 2017
for i in range(60, 80):
	# Appending the scaled version of all of our days
	X_test.append(inputs[i-60:i, 0])

# We need to convert X_test to numpy arrays so it'll be accepted by the RNN
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the prediction!
predicted_stock_price = regressor.predict(X_test)

# Un-scale the prediction
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Compute RMSE
import math
from sklearn.metrics import mean_squared_error

absolute_rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

relative_rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price)) / 800

print "The absolute RMSE is: " + str(absolute_rmse)
print "The relative RMSE is: " + str(relative_rmse)

#####################
### Visualization ###
#####################

plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# How to Improve the Model
# (1) Get more training data
# (2) Increase the number of timesteps
# (3) Add other stock tickers into the mix if there may be some correlation with the Google stock price
# (4) Add more LSTM layers
# (5) Add more neurons (units) to the layers

# Tuning the RNN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense

# def build_classifier(optimizer):
#     classifier = Sequential()
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#     classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#     classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#     classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['neg_mean_squared_error'])
#     return classifier

# classifier = KerasClassifier(build_fn = build_classifier)
# parameters = {'batch_size': [25, 32],
#               'epochs': [100, 500],
#               'optimizer': ['adam', 'rmsprop']}
# grid_search = GridSearchCV(estimator = classifier,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10)
# grid_search = grid_search.fit(X_train, y_train)
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_