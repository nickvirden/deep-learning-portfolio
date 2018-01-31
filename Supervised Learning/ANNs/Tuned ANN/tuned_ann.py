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

# Part 4 - Evaluating, Improving, and Tuning the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense # creates the hidden layers in Keras
from keras.layers import Dropout

# classifier = Sequential()
# classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
# # Dropout
# # p => % of neurons you wanna disable
# #   => start with 0.1 and step upward by 0.1
# classifier.add(Dropout(p=0.1))
# classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
# classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# classifier.fit(X_train, y_train, batch_size=10, epochs=10)

# Evaluate the ANN
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score

# def build_classifier():
# 	classifier = Sequential()
# 	classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
# 	classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
# 	classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# 	classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# 	return classifier

# classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
# # Goes over 10 folds by fitting the classifier on the training sets
# accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)

# print accuracies.mean()
# print accuracies.std()

# Improve ANN
# Implement Dropout Regularization technique to combat 

# Tune ANN
# Implement GridSearchCV
# Evaluate the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
	classifier = Sequential()
	classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
	classifier.add(Dropout(0.2))
	classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
	classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
	classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	return classifier

classifier = KerasClassifier(build_fn=build_classifier)

# Keras docs recommend rmsprop for recurrent neural networks
parameters = {
	'batch_size': [21, 23, 25, 27, 29],
	'epochs': [490, 500, 510],
	'optimizer': ['adam', 'rmsprop']
}

grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print "Best Parameters"
print best_parameters

print "Best Accuracy"
print best_accuracy

# Without Dropout
# Best parameters seems to be batch_size=25, epochs=500 with an indifference for the optimizer
# Accuracy converges to ~85%

# With Dropout
# Best parameters seems to be batch_size=25, epochs=500 with an indifference for the optimizer
# Accuracy converges to ~85%