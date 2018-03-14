# Install PyTorch (http://pytorch.org/)
# Had to install from source: https://github.com/pytorch/pytorch#from-source
# Datasets retrived from https://grouplens.org/datasets/movielens/

##########################
### BOLTZMANN MACHINES ###
##########################

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Import the dataset
# Here, we specify the separator as :: because if movies have a comma in their
# title, it will show up twice or in a weird format
# We specify the encoding as latin-1 to ensure that special characters are
# converted correctly
movies = pd.read_csv('ml-1m/movies.dat',
                     sep='::',
                     header=None,
                     engine='python',
                     encoding='latin-1')

# User ID || Gender || Age || Job Code ID || Zip Code
users = pd.read_csv('ml-1m/users.dat',
                     sep='::',
                     header=None,
                     engine='python',
                     encoding='latin-1')

# Foreign Key User ID || Movie IDs || Ratings (1 - 5) || Timestamps
ratings = pd.read_csv('ml-1m/ratings.dat',
                     sep='::',
                     header=None,
                     engine='python',
                     encoding='latin-1')

# Prepare the training set and test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
# In order to manipulate the data set using PyTorch tensors,
# we must convert it into a numpy array
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Get the number of users and movies
# Step 1 - Make matrices out of training and test set
# Step 2 - If the user rated the movie, we'll denote it with 1, otherwise 0

# Get all the users in the training set and find the highest users ID
number_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
number_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) 

# Convert data into array with users in lines and movies in columns
# Why? Because the RBM expects data in this format

# Creates a list of lists => list of all the movies a user has rated
# Outer list represents a user; inner list represents movies
def convert(data):
    new_data = []
    
    # We start at 1 because the user IDs start at 1
    for id_users in range(1, number_users + 1):
        # Create a mask that only gets the current user's data
        user_mask = data[:, 0] == id_users
        # Get the whole column of the movie IDs
        id_movies = data[:, 1][user_mask]
        # Get ratings given by the user
        id_ratings = data[:, 2][user_mask]
        # We want to get a 1 if a user rated the movie and a 0 if they didn't
        # We need a list of 1682 elements (i.e. the number of movies), then we will replace the 0 by the actual user rating
        ratings = np.zeros(number_movies)
        # Replace all the zeroes by the actual ratings
        ratings[id_movies - 1] = id_ratings
        # Add ratings list to the 'new_data' list
        new_data.append(list(ratings))
    
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert the data into Torch tensors
# TENSOR => Multidimensional matrix that contain elements of a single data type
# Using PyTorch, we create Torch Tensors, which are much more efficient than numpy arrays
# PyTorch's advantage over Tensorflow when building an Auto-Encoder from scratch is that it is simpler and quicker

# FloatTensor class expects a list of lists, so that's why the earlier conversion was necessary
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Convert the ratings into binary ratings where 1 is LIKED and 0 is DISLIKED
# First, we'll replace all the 0s with -1 because all the ratings with 0 corresponded to the movies that were not rated by a user
# Take all the values of the training set that have a zero and replace them with -1
training_set[training_set == 0] = -1
# Since PyTorch Tensors do not respond to the OR operator, we have to use two different expressions
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
# Perform a similar operation for positive ratings
training_set[training_set >= 3] = 1
# Do the same for the test set
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Architect the RBM
# CLASS => A model of something we want to build // an ensemble of instructions
class RBM():
    # We always have to start with an init function when creating a class
    # self => default, compulsory argument
    # num_vis_nodes => number of visible nodes
    # num_hidden_nodes => number of hidden nodes
    def __init__(self, num_vis_nodes, num_hidden_nodes):
        # We are going to initialize the parameters for our class
        # All of the params that we will optimize during our training
        # W => weights | all the params of the probabilities of the visible nodes given the hidden nodes
        #   => initialized in a matrix (Tensor) of size num_vis_nodes * num_hidden_nodes in a normal distribution
        self.W = torch.randn(num_hidden_nodes, num_vis_nodes)
        # Initialize the bias for hidden nodes given the visible nodes
        self.a = torch.randn(1, num_hidden_nodes)
        # Initialize the bias for visible nodes given the hidden nodes
        self.b = torch.randn(1, num_vis_nodes)
    
    # Sample the hidden nodes according to the probabilities that a node is a hidden node given a visible node
    # The Sigmoid Activation Function corresponds to this probability
    # Purpose: During training, we'll approximate the Ludd likelihood gradient through Gibbs sampling
    # Samples the activations of these hidden nodes (i.e. whether the hidden node is activated | equals 1) given the visible node
    def sample_h(self, num_visible_neurons):
        # Compute the hidden neuron equals 1 given the value of the visible node
        # The probability of h given v is the Sigmoid Activation Function applied to W (the vector of weights) * num_visible_neurons + the bias of the num_hidden_nodes
        w_num_visible_neurons = torch.mm(num_visible_neurons, self.W.t())
        # expand_as ensures that the bias is applied evenly to each mini-batch
        activation = w_num_visible_neurons + self.a.expand_as(w_num_visible_neurons)
        # Probability that hidden node is activated given the visible node
        # Ex. user that only likes dramatic movies
        # If we know that the user rates drama movies high, which in this case are all the visible nodes = 1, 
        # then the probability of h, the hidden node that has characteristics of a good drama movie, 
        # then the probability that that movie will be recommended will be really high
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    # We want to return the probabilities that the visible nodes equal 1 given the hidden nodes
    def sample_v(self, num_hidden_neurons):
        # W is not transposed because here because it deals with visible nodes
        w_num_hidden_neurons = torch.mm(num_hidden_neurons, self.W)
        activation = w_num_hidden_neurons + self.b.expand_as(w_num_hidden_neurons)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    # We want to minimize the energy it takes to calculate the gradient, so we're
    # going to approximate it instead of calculate it directly
    # Contrastive Divergence => Utilize Gibbs Sampling to select a number of hidden visible nodes k times
    # input_vector => contains all of the movie ratings by one user -- will loop over all users
    # visible_nodes_obtained_after_k_samplings => nodes obtained from k Gibbs Samplings
    # ph0 => the vector of probabilities that at the first iteration the hidden nodes equal 1 given the values of the input_vector
    # phk => the vector of probabilities of the hidden nodes after k samplings given the values of the visible nodes vk
    def train(self, input_vector, visible_nodes_obtained_after_k_samplings, ph, phk):
        # Update weights using torch matrix multiplication
        self.W += torch.mm(input_vector.t(), ph) - torch.mm(visible_nodes_obtained_after_k_samplings.t(), phk)
        self.b += torch.sum((input_vector - visible_nodes_obtained_after_k_samplings), 0)
        self.a += torch.sum((ph - phk))

num_vis_nodes = len(training_set[0])
# The number of hidden nodes is tuneable, so we select 100 here to test
num_hidden_nodes = 100
batch_size = 100

# Initialize RBM
rbm = RBM(num_vis_nodes, num_hidden_nodes)

# Train the RBM
# Since we have a binary outcome, convergence will be reach relatively quickly
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    # For any deep learning algorithm, we need a loss function
    # Measure the difference between real and predicted ratings
    # RMSE
    train_loss = 0
    # Use counter to normalize the loss // Float because division
    s = 0.
    
    # Loop through all of the users
    for id_user in range(0, number_users - batch_size, batch_size):
        # Get vector of ratings that already exist
        visible_nodes_obtained = training_set[id_user:id_user + batch_size]
        # Since the target is the same as the input at the beginning, we have the same values
        # The original ratings at the beginning
        v0 = training_set[id_user:id_user + batch_size]
        # Initial Probabilities
        # ph0,_ only gets the first element
        ph0,_ = rbm.sample_h(v0)
        
        for k in range(10):
            
        
        
        
        
        
        
        
        
        
        
