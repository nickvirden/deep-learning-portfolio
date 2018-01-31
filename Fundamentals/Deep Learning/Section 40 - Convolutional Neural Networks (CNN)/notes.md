# Step 1A: Convolution Operation
Paper that started convolutional neural networks: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

Paper that's aimed at beginners in CNNs: http://cs.nju.edu.cn/wujx/paper/CNN.pdf

## Steps to Convolution
**Equation**
Input Image * Feature Detector = Feature Map

**Inputs**
- Input image represented as 1s and 0s
- Feature Detector / Kernel / Filter -> typically 3 x 3 matrix of a particular feature

**Steps**
1) Start in top left corner of Input Image and overlay Feature Detector
- Element-wise multiply each element of the matricies to see how similar the piece of the Input Image is to the Feature Detector
2) Step through entire image until Feature Detector has covered every area of the Input Image, producing a complete Feature Map

**Output**
- Feature Map / Convolved Feature / Convolved Map

**Feature Detector**
We lose data when we use a Feature Detector, but we're trying to discern features from a photo, so we necessarily use information

# Step 1B: Rectified Linear Unit (ReLU) Function

We want to increase non-linearity in our network because images are non-linear themselves
Rectifier function turns all the negative values into 0. You break the linearity by removing all negative values.

Understanding CNNs with a Mathematical Model: https://arxiv.org/pdf/1609.04112.pdf
Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification: https://arxiv.org/pdf/1502.01852.pdf

# Step 2: Max Pooling

**Spatial Invariance:** Computer doesn't care where features are so it can recognize different variants of the same type of object

Max pooling is just one type of pooling
- Others include Mean Pooling and Sum Pooling

**Max Pooling** => Step through Feature Map and record only Max values in the 2 x 2 matrix you overlay on the feature map

Max Pooling features removes any noise due to distortion, overfitting

Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition: http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf

# Step 3: Flattening

This is literally just taking the pooled / downsampled layers and putting them into a 2D vector with on column and many rows to make them input nodes for the ANN

# Step 4: Full Connection

**Hidden Layer**
In other ANNs, a layer does not necessarily have to be connected to every node from the previous layer

**Fully Connected Layers** 
In CNNs, a layer necessarily has to be connected to every node in the previous layer of the network


# Additional Reading
https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html

# Softmax & Cross-Entropy

## Softmax
**Q:** If the output neurons are not connected to each other, why do their probabilities add up to 1?
**A:** Applying the Softmax function to the output values produces individual values between 0 and 1 that add up to 1

## Cross-Entropy

**Cross-Entropy:** a loss function that we use to minimize the error in a given NN
- Since cross-entropy has a logarithm in it, the difference in performance is more apparent since logarithms amplify super small differences. 

**EXAMPLE** 
If you improve performance from 0.000001 and 0.001 have small absolute difference, but a logarithm would show a 1000x improvement to guide your model better. Mean-Squared difference would not show this.

**Caveat:** If you're doing regression, Mean Squared Error is great. But, Cross-Entropy is good for CNNs.

**Recommendation** => Check out Jeffrey Hinton on YouTube

A Friendly Introduction to Cross-Entropy Loss (2016) => https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/