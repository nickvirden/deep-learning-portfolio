# Artificial Neural Networks

# Install Tensorflow
# pip install --upgrade tensorflow

# Installing Keras
# pip install --upgrade keras
# Based on Theano and Tensorflow
# Simplifies building neural networks (i.e. requires less lines of code to create a neural network)

# Part 1 - Data Preprocessing

# Import the libraries
import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode the Country values
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])

# Encode the Gender values
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])

# Since there is no ordinal relationship between the country names, we avoid the dummy variable trap by using One Hot Encoding
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()

# Columns 1, 2, and 3 are the Country dummy variables
# We want only two of the dummy variables at a time, so we select the second column onward to get rid of the first dummy variable
X = X[:, 1:]

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Making the ANN

# Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense # creates the hidden layers in Keras

# Initialize the ANN
classifier = Sequential()

# Add the input layer and the first hidden layer
# .add() adds the hidden layer
# output_dim => number of nodes in the hidden layer
# -- No rule of thumb for optimal number in hidden layer; Tip: take AVG(SUM(input nodes, output nodes))
# -- Best way to validate results is to use k-fold cross-validation (Part 10)
# kernel_initializer => initialization function that initializes weights close to 0
# activation => activation function in the hidden layer, which in this case is the Rectifier (Rectified Linear Unit) function
# input_dim => number of independent variables
#           => every subsequent layer doesn't need this parameter because this layer is here
classifier.add(Dense(output_dim=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Add the second hidden layer
classifier.add(Dense(output_dim=6, kernel_initializer='uniform', activation='relu'))

# Add the output layer
# output_dim => Since we have an output with a binary outcome, we only need 1 output
#            => If we had 3 categorical variables as an outcome, we'd have to change it to 3
# activation => sigmoid because sigmoid projects the value to either 0 or 1 based on if the probability is > or < 0.5
#            => we would want 'softmax' here if we had categorical variables with multiple outcomes
classifier.add(Dense(output_dim=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN
# optimizer => the algorithm you want to use to find the optimal weights in the NN
#           => stochastic gradient descent -> we use Adam in this case
# loss => stochastic gradient descent is based on a loss function that you have to optimize to find the optimal weights
#      => binary outcome --> binary_crossentropy // multiple outcomes --> categorical_crossentropy
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Make predictions!

# Predict the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

print new_prediction

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print cm