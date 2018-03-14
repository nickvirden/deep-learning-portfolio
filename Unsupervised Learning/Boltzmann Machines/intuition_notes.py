##########################
### BOLTZMANN MACHINES ###
##########################

# A Boltzmann Machine describes a model
# Stochastic / Generative model
# Learning through good examples, a Boltzmann Machine can model and determine ideal and abnormal states
# All of the nodes are interconnected so the nodes can share information to find potential connections in the data

###########################
### Energy-Based Models ###
###########################

# Energy-Based Models describe the probability of something being in a certain state
# The Boltzmann Distribution formula basically says that the state of a system is inversely proportional to its 
# probability being in that state

#####################################
### Restricted Boltzmann Machines ###
#####################################

# The RBM uses its hidden layer to reconstruct the inputs and then make predictions based on which nodes are connected
# to the inputs about which we are seeking data or have no data

##############################
### Contrastive Divergence ###
##############################

# Q: Why are the reconstructed nodes not the same as the input nodes?
# A: The input nodes are not initially interconnected, hence "restricted", so the hidden node weights are built
# from all of the input nodes. Upon reconstruction, the hidden nodes use all of the weight information, not just singular
# weight information, so we get different reconstructed nodes

# Gibb's Sampling: passing information between input nodes and hidden nodes until at some point the input nodes
# are reconstructed exactly as they were passed in

# Hinton's Shortcut: It's sufficient to take just the first two "passes" of the Gibb's Sampling to figure out how
# we need to adjust the weights in the energy curve to get the lowest possible energy state for our system

############################
### Deep Belief Networks ###
############################

# A stack of several RBMs
# It's key to ensure that directionality is established for all of the layers, except the top two, toward
# the input layer

### Greedy Layer-Wise Training => Train the RBM layer by layer and then establish directionality
### Wake-Sleep Algorithm => Train all the way up a series of connections, then all the way down the same series of connections

###############################
### Deep Boltzmann Machines ###
###############################

# No directionality as compared to DBNs