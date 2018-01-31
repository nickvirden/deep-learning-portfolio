# Hierarchical Clustering

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# Get values for regression
X = dataset.iloc[:, [3,4]].values

# Import scipy to create a dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch

# Create the dendrogram
# method='ward' => tries to minimize the number of clusters
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

# Display the dendrogram
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean Distances')
# plt.show()

# Fit hierarchical clustering to the dataset
from sklear.cluster import AgglomerativeClustering

# n_clusters => we chose 5 because that's the number we deteremined from the dendrogram
# linkage='ward' => tries to minimize number of clusters
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

# Fit the algorithm to X and predict the number of clusters
y_hc = hc.fit_predict(X)

# Visualize clusters
# Parameters scatter(x coordinates, y coordinates, size, color of points, label for set)
# y_hc==0 means that we want the observation that belongs to cluster 1
# 0 => we want the first column of our data X, which is our x coordinate
# 1 => second column of our data X, which is our y coordinate
# plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', label='Careful')
# plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', label='Standard')
# plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', label='Target')
# plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', label='Careless')
# plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', label='Sensible')
# plt.title('Cluster of Clients')
# plt.xlabel('Annual Income ($1000s)')
# plt.ylabel('Spending Score (1 - 100)')
# plt.legend()
# plt.show()