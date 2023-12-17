# In the following neural network that has a hidden layer, perform the backpropagation algorithm twice and update the weight vectors in the network with the gradient reduction method along with momentum. The initial values are given in the figure. Use the mean squared error function as the cost function. Sigmoid function functions.
# (the neural network has an input with size three (x1, x2, x3), a hidden layer with two neurons that is connected and to the a3 layer)
# input:
# x1 = 1, x2 = 0, x3 = 1
# weights of hidden layer:
# w11 = 0.2, w12 = -0.3, w21 = 0.4, w22 = 0.1, w31 = -0.5, w32 = -0.2, b1 = -0.4, b2 = 0.2
# a3 layer:
# u1 = -0.3, u2= -0.2, b3 = 0.1, 
# target:
# Y = 1
# learning rate = alpha = 0.9


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Initialize weights and biases
w11, w12, w21, w22, w31, w32 = 0.2, -0.3, 0.4, 0.1, -0.5, -0.2
b1, b2 = -0.4, 0.2
u1, u2 = -0.3, -0.2
b3 = 0.1

# Input
x1, x2, x3 = 1, 0, 1

# Target
target = 1

# Learning rate and momentum
alpha = 0.9
beta = 0.8

# Forward pass
z1 = w11 * x1 + w21 * x2 + w31 * x3 + b1
a1 = 1 / (1 + np.exp(-z1))

z2 = w12 * x1 + w22 * x2 + w32 * x3 + b2
a2 = 1 / (1 + np.exp(-z2))

z3 = u1 * a1 + u2 * a2 + b3
a3 = 1 / (1 + np.exp(-z3))

# Mean squared error loss
loss = 0.5 * (a3 - target) ** 2

# Backward pass
delta_a3 = (a3 - target) * a3 * (1 - a3)

delta_a1 = (u1 * delta_a3) * a1 * (1 - a1)
delta_a2 = (u2 * delta_a3) * a2 * (1 - a2)

# Update weights and biases with momentum
w11 -= alpha * (delta_a1 * x1) + beta * w11
w12 -= alpha * (delta_a2 * x1) + beta * w12
w21 -= alpha * (delta_a1 * x2) + beta * w21
w22 -= alpha * (delta_a2 * x2) + beta * w22
w31 -= alpha * (delta_a1 * x3) + beta * w31
w32 -= alpha * (delta_a2 * x3) + beta * w32

b1 -= alpha * delta_a1 + beta * b1
b2 -= alpha * delta_a2 + beta * b2

u1 -= alpha * (delta_a3 * a1) + beta * u1
u2 -= alpha * (delta_a3 * a2) + beta * u2
b3 -= alpha * delta_a3 + beta * b3
