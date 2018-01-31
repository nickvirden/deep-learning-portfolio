# Thompson Sampling

# We are creating distributions of where we *think* the mean of the distribution might lie
# Creating an auxillary mechanism to solve the problem
# We are NOT trying to guess the actual returns

# Thompson Sampling pulls a value out of each distribution
# From this sampling, we have generated our own bandit configuration

# CHARACTERISTICS
# Probabilistic => generating values based on probability to get closer and closer to the real mean
# Accomodates Delayed Feedback
# Better Empirical Evidence

# By taking the maximum draw, we are estimating the highest probability of success -- corresponds to one specific ad at each round

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implement Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
number_of_rewards_1 = [0] * d
number_of_rewards_0 = [0] * d
total_reward = 0

for n in range(0, N):

	# Initialize the ad that we will show during the round
    ad = 0

    # Maximum random draw
    max_random = 0

    for i in range(0, d):

    	# Corresponds to different random draws
    	random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)

        if random_beta > max_random:
            max_random = random_beta
            ad = i

    ads_selected.append(ad)

    reward = dataset.values[n, ad]

    if reward == 1:
    	number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
    	number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
    
    total_reward = total_reward + reward

print total_reward

# Visualising the results
# plt.hist(ads_selected)
# plt.title('Histogram of ads selections')
# plt.xlabel('Ads')
# plt.ylabel('Number of times each ad was selected')
# plt.show()