# Logistic Regression

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Split the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Principal Component Analysis (PCA)
from sklearn.decomposition import PCA

# Number of extracted features you wanna get that will explain the variance the most
# We want to get two principal components, so we input None because we will create a vector called explains_variance that explains the variance the most out of all the principal components
pca = PCA(n_components=None)

# Apply fit_transform so the PCA object can see how the training set is structured and extract some new independent variables that explain the most variance
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Analyze the cumulative explained variance of each of the principal components
explained_variance = pca.explained_variance_ratio_

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fit Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression

# Create classifier
classifier = LogisticRegression(random_state=0)

# Fit classifier to Training set data
# The classifier will learn the correlations between X_train and y_train. Then, by learning those correlations, it will be able to use what it learned to predict new observations
classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier.predict(X_test)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix

# We use y_test because we need to compare matrices that are the same size
# The results are [[65, 3], [8, 24]], which means:
# For "Purchased", 65 results were predicted correctly, while 3 were predicted incorrectly
# For "Not Purchased", 24 results were predicted correctly, while 8 were predicted incorrectly
cm = confusion_matrix(y_test, y_pred)

print cm

# Visualize the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_train, y_train

# Create a graph that starts minimum - 1 values and ends at the maximum + 1 values with a high resolution to make the points look continuous
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# # Red is Did Not Purchase
# # Green is Did Purchased
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()
# plt.show()

# Visualize the Test set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = X_test, y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()
# plt.show()