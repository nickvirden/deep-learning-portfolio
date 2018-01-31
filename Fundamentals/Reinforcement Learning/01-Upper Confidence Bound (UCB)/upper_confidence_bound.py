# Upper Confidence Bound

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implent UCB
# NO LIBRARY TO DO THIS :O
import math
N = 10000
d = 10
# Number of times an ad was selected by the algorithm, which should sum to N
numbers_of_selections = [0] * d # Creates a vector with 10 zeros that will keep track of how many times each ad is selected
# If we select the ad that they're going to click on, then it increments the sum_of_rewards[index] by 1
sum_of_rewards = [0] * d
total_reward = 0

# GOAL: Select the ad i that has the maximum UCB
# => This means that we need to create a huge vector that shows which version of the ad was selected at each round
ads_selected = []

# Average reward of ad i up to round n
# Go over all of the rounds
for n in range(0, N):

	# Keep track of ad with highest upper bound in each round
	ad = 0

	# This max upper bound needs to be calculated in order to pick the ad with the highest upper bound at each round and then add it to the ads_selected list
	max_upper_bound = 0

	# Go over all of the ads in a specific round
	for i in range(0, d):

		# If the ad version, i, was selected at least once, then we will use this strategy
		if numbers_of_selections[i] > 0:
			# Average Reward
			average_reward = sum_of_rewards[i] / numbers_of_selections[i]

			# numbers_of_selections[i] => number of times the ad version i was selected up to round n
			delta_i = math.sqrt((3/2) * (math.log(n + 1) / numbers_of_selections[i]))

			# UCB
			upper_bound = average_reward + delta_i
		# Otherwise
		else:
			upper_bound = 1e400
		# If the upper bound is greater than the current highest upper bound
		if upper_bound > max_upper_bound:
			# Keep track of the ad that has the max upper bound
			ad = i

			# Update max upper bound every time a better one emerges
			max_upper_bound = upper_bound

	# Add each ad to the list of selected ads
	ads_selected.append(ad)

	# Increment the count for whichever ad was selected
	numbers_of_selections[ad] = numbers_of_selections[ad] + 1

	# The real reward at round n
	reward = dataset.values[n, ad]

	# Sum of rewards for a particular ads
	sum_of_rewards[ad] = sum_of_rewards[ad] + reward

	total_reward = total_reward + reward

print "Total reward: " + str(total_reward)

# Visualize the data
# plt.hist(ads_selected)
# plt.title('Histogram of ads selections')
# plt.xlabel('Ads')
# plt.ylabel('Number of times each ad was selected')
# plt.show()