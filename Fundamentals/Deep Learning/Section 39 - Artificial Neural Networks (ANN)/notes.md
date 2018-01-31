# Key Functions
## Threshold Function
- If value < 0, returns 0
- Otherwise, if value >= 0, returns 1

## Sigmoid Function
1 / (1 + e^-x)
Bounded on the y-axis between 0 and 1

## Rectifier Function**
- If value < 0, returns 0
- If value >= 0, returns a value between 0 and 1 based on the slope of the line from (0, 0) to (x, 1)

** One of the most used functions in ANN

## Hyperbolic Tangent (tanh)
**Equation** => (1 - e^-2x) / (1 + e^-2x)
Bounded on the y-axis between -1 and 1

## (Most Common) **Cost Function** 
SUM((y-hat - y)^2 / 2)
- Tells us the error in the calculation by the NN

# Applying Functions to Neural Networks
Assuming the dependent variable (DV), y, is binary (returns 0 or 1), which function would you use?

## Option 1 - Threshold Function
## Option 2 - Sigmoid Function

Common to apply **Recitifier** function in the **hidden** layers and then the **Sigmoid** function in the **output** layers

Once the Cost Function is calculated, it sends feedback to the NN, adjusts the weights of the input values, and then takes another input

# How to Minimize the Cost Function
**Curse of Dimensionality** => Explains that checking every combination possible for the cost function would be so time intensive that it's better to find some other method to determine the cost function's minimum

## Batch Gradient Descent / Stochastic Gradient Descenct
- Requires that function is convex, but obviously this is not always the case

### Batch Gradient Descent
- Update weights of NN inputs after iterating over two or more calculations
**Benefit**
- Deterministic algorithm


### Stochastic Gradient Descent
- Update weights of NN inputs after every calculation
- Helps avoid local minimums, versus overall global minimum

**Benefits**
- Has much higher fluctuations, so it's much more likely to find the global minimum
- It's faster than Batch Gradient Descent
- Stochastic (Probabilistic) algorithm

#### Training the ANN with Stochastic Gradient Descent
**Step 1:** Randomly initialize weights with small number close to 0
**Step 2:** Input first observation of dataset into input layer; each feature is one input node
**Step 3:** Forward-Propagation: left-to-right activation of neurons based on the weights
**Step 4:** Compare predicted result to actual result
**Step 5:** Back-Propagation: right-to-left update of weights based on how responsible they are for the error
**Step 5A:** Reinforcement Learning => update weight after each observation
**Step 5B:** Batch Learning => update weight after a batch of observations

**Soft Intro to the Math behind Stochastic Gradient Descent**
https://iamtrask.github.io/2015/07/27/python-network-part2/
http://neuralnetworksanddeeplearning.com/chap2.html

# Key Terminology
**Single-Layer Feed Forward Neural Network** - a NN with just one hidden layer; also called a **Perceptron**

**Backpropogation**
- Simply because of the way the algorithm is structured, you are able to adjust all of the weights at the same time
- So you know which part of the error (Cost) each of your weights is responsible for
http://neuralnetworksanddeeplearning.com/chap2.html