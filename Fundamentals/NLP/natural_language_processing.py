# Natural Language Processing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
# quoting = 3 => requires that quotes have at least 3 in a row, or else quotes are ignored
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Clean up the text
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 1000):

	review = re.sub('[^A-Za-z]', ' ', dataset['Review'][i])
	review = review.lower()
	review = review.split()
	ps = PorterStemmer()

	# Create a list comprehension that iterates over all the words in a review, removes the stop words, and creates a list of each set of non-stop-words in a review
	# Sets are faster to iterate over than lists, so set is a performance optimization
	review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

	review = " ".join(review)
	corpus.append(review)

# Create the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# Count Vectorizer does have its own parameters built in to clean text, but doing it manually as above allows more options
# max_features => the number of words you want to keep
cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Naive Bayes Implementation
# Feature scaling is unnecessary because all the values are very close to each other

# Split the dataset into training and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Import and instantiate a classifier
# NAIVE BAYES
# from sklearn.naive_bayes import GaussianNB
# classifier = GaussianNB()

# DECISION TREE
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

# RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=0)

# Fit the classifier to the training set
classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred = classifier.predict(X_test)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print "----------- Confusion Matrix -----------"
print cm
print "----------------------------------------"

# CLASSIFIER SCORING

# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# Score = 2 * Precision * Recall / (Precision + Recall)

# Naive Bayes correctly identifies 55 positive reviews and 91 negative reviews; and incorrectly identifies 42 reviews as positive and 12 reviews as negative

# Accuracy = (55 + 91) / 200 = 73%
# Precision = 55 / (55 + 42) = 56.7%
# Recall = 55 / (55 + 12) = 82.1%
# Score = (2 * .567 * .821) / (.567 + .821) = 67.07%

# Decision Tree correctly identifies 74 positive reviews and 68 negative reviews; and incorrectly identifies 23 reviews as positive and 35 reviews as negative

# Accuracy = (74 + 68) / 200 = 71%
# Precision = 74 / (74 + 23) = 76.2%
# Recall = 74 / (74 + 35) = 67.9%
# Score = (2 * .762 * .679) / (.762 + .679) = 71.81%

# Random Forest (10) correctly identifies 87 positive reviews and 57 negative reviews; and incorrectly identifies 10 reviews as positive and 46 reviews as negative

# Accuracy = (87 + 57) / 200 = 72%
# Precision = 87 / (87 + 10) = 89.6%
# Recall = 87 / (87 + 46) = 65.4%
# Score = (2 * .896 * .654) / (.896 + .654) = 75.61%

# Random Forest at 100 and even 1000 estimators nearly converge on the results provided by Random Forest at 10 estimators