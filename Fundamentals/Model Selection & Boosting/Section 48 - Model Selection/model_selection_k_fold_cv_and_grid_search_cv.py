# k-Fold Cross Validation

# Parameters => Parameters that model learned
# Hyperparameters => parameters that we chose, such as kernal, number of nodes, etc.

# k-Fold Cross Validation will help tackle the variance problem and introduce a new metric for testing the performance of an algorithm
# => Takes the training set, divides it into k sets, then trains on k-1 sets and makes a prediction on the third
# => There is always a bias-variance trade-off

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Make confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print cm

# Apply k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
accuracies_mean = accuracies.mean()
accuracies_st_dev = accuracies.std()

print accuracies_mean
print accuracies_st_dev