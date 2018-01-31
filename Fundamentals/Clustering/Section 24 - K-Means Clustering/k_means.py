# Data Preprocessing Template

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
# y = dataset.iloc[:, 3].values

# Use the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans

# Store the 10 different Within-Cluster Sum of Squares of our 10 different cluster initializations
wcss = []

# Iterate over range of values
for i in range(1, 11):
	# Create instance of KMeans class
	# n_clusters -> number of clusters
	# init -> use K-Means++ initialization method so as not to fall into Random Initialization Trap
	# max_iter -> maximum number of iterations there can be to find a final cluster / default: 300
	# n_init -> number of times KMeans cluster will be run with different initial centroids
	kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
	# Fit the KMeans algorithm to the data
	kmeans.fit(X)
	# WCSS => Inertia
	wcss.append(kmeans.inertia_)

# Plot the data
# plt.plot(range(1, 11), wcss)
# plt.title('The Elbow Method')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS (Inertia)')
# plt.show()

# Apply K-Means to the Mall Dataset
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit K-Means to the dataset
# Using fit_predict because it returns which cluster each observation belongs to
# For every single client of our dataset, the fit_predict method will tell us which cluster they belong to
y_kmeans = kmeans.fit_predict(X)

print y_kmeans
print X

# Visualize the clusters
# Parameters scatter(x coordinates, y coordinates, size, color of points, label for set)
# y_kmeans==0 means that we want the observation that belongs to cluster 1
# 0 => we want the first column of our data X, which is our x coordinate
# 1 => second column of our data X, which is our y coordinate
# plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label='Cluster 1')
# plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label='Cluster 2')
# plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label='Cluster 3')
# plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label='Cluster 4')
# plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label='Cluster 5')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
# plt.title('Cluster of Clients')
# plt.xlabel('Annual Income ($1000s)')
# plt.ylabel('Spending Score (1 - 100)')
# plt.legend()
# plt.show()