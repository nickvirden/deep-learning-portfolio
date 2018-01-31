# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Suppresses automatic format guessing by numpy
# I set this option because it changes numbers into scientific notation after they get disproportionately large
np.set_printoptions(suppress=True)

# Import the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Take care of missing data
from sklearn.preprocessing import Imputer

# Imputer by default replaces NaN values with the mean of the column
imputer = Imputer()

# Fits Imputer object to Matrix X
imputer = imputer.fit(X[:, 1:3])

# Replace the data in the second and third columns with the mean of their column values
X[:, 1:3] = imputer.transform(X[:, 1:3])

# View the filled in variables
# print X[:, 1:3]

# Encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Create new instance of label encoder for the independent variables
label_encoder_X = LabelEncoder()

# Encodes categorical variables in the first column
# In this case, the first column is the countries
# The only problem here is that if we just do encoding on one column, the program will think, for example, France (1) is greater than Germany (0), which makes no sense, so we have to use dummy variables
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

# Print out (incorrect) encoded Matrix X
# print X

# OneHotEncoder will treat all the columns as categorical variables by default, so we pass an array with the indices into the categorical_features paramter to encode only that column
one_hot_encoder = OneHotEncoder(categorical_features=[0])

# Returns three categorical columns - one for each country - as a numpy array (matrix)
X = one_hot_encoder.fit_transform(X).toarray()

# Print variables rounded to 2 decimal places
# print np.around(X[:, :-1], decimals=2)

# Create new instance of label encoder for the dependent variable
label_encoder_y = LabelEncoder()

# Label encode dependent variable "Purchased" 
y = label_encoder_y.fit_transform(y)

# Split the dataset into the Training and Test sets
from sklearn.cross_validation import train_test_split

# Create Training and Test sets that are associated with each other
# As a general rule of thumb, the test size should be somewhere between 20% and 30% of the dataset, and certainly no more than 40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print np.around(X_test, decimals=2)
# print np.around(X_train, decimals=2)
# print np.around(y_test, decimals=2)
# print np.around(y_train, decimals=2)

# The last step in data preprocessing is normalizing the data so no one feature dominates the fit of the dataset
# For example, salary is in $1,000s, whereas age is in 10s, so we need to bring the values to an equal scale in order to get a proper fit

# TWO METHODS OF FEATURE SCALING
# Standardization => (x - mean(data)) / st-dev(data)
# Normalization => (x - min(data)) / (max(data) - min(data))

from sklearn.preprocessing import StandardScaler

# Scale X
sc_X = StandardScaler()

# Fit and transform Training set
X_train = sc_X.fit_transform(X_train)

# Only need to transform Test set because we already fit the X Matrix
X_test = sc_X.transform(X_test)

# Sometimes y will need to be fitted, but in this case, it's only 0 and 1, so we don't need to scale it

print np.around(X_test, decimals=3)
print np.around(X_train, decimals=3)
print np.around(y_test, decimals=3)
print np.around(y_train, decimals=3)