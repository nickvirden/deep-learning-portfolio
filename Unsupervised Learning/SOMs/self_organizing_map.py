import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data set
dataset = pd.read_csv('Credit_Card_Applications.csv')

# PROBLEM: How do we determine what an outlier is?
# Mean Inter-neuron Distance (MID) => For each neuron, we're going to compute the Euclidean distance between the neuron and the neurons in its "neighborhood", which we define manually

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Unsupervised Deep Learning means that not dependent variable is considered

# Feature Scaling
# Our SOM has high dimensionality, so we need to reduce it to make it more efficient

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))

# Get the normalized version of X
X = sc.fit_transform(X)

# Train the SOM
from minisom import MiniSom

# learning_rate => the higher the rate, the faster the model will converge
# x & y => determines the size of the map that we want
# input_len => number of features (columns) that we've highlighted in the dataset
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)

# This chooses the random nodes that will be selected during each iterations
som.random_weights_init(X)

# Number of times we want to initialize random iterations
som.train_random(data=X, num_iterations=100)