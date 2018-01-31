# Apriori

# Ex. Recommender Engine for Movies
# Support => Number of People who have Seen Movie X / Total Number of People in Sample Population
# Confidence => Number of People who have Seen Movie X & Y / Number of People who have Seen Movie X
# Lift => Confidence / Support
# -- If LIFT is greater than 1, it means that if you ask someone "Hey do you like movie X?" and then recommend them movie Y, you have an equal likelihood that they like movie Y whether you chose them out of this sub-population or the entire population. Anything above 1 means there is a greater likelihood that someone will like movie Y if they like movie X already.

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Put all the different transactions in an array
transactions = []

# Loop over all the transactions
for i in range(0, 7501):

	# For each transaction, create a list of all the items in a transaction
	# We must convert the values to a string because that is simply what the apriori function is expecting
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Train Apriori on the dataset
from apyori import apriori

# Creates a set of rules based on the transactions
# IMPORTANT NOTES
# min_support => number of items purchased * days in period / total number of transactions
# min_confidence -- If confidence is too high, then only obvious rules emerge since CONFIDENCE is a measure of how often items are associated
# OBVIOUS RULE IF CONFIDENCE IS TOO HIGH (e.g. 80%) => a product that occurs more times overall will result in being a rule simply because it is popular, not because it is well associated with other transactions
# ANOTHER EXAMPLE => Hot in south of France in summer, so people will buy water. Also, French people love eggs in general. Since both items occur in the same transaction seasonally, it could be drawn from the data that eggs and water go together when in fact there are other underlying patterns
# min_confidence set to 0.2 ... (fill in)
# min_lift => 
# min_length => rules composed of at least n values
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
# Rules are already sorted by their relevance, so no sorting from greatest to least needed
# Rules are sorted by a combination of their support, confidence, and lift via the apriori function
results = list(rules)