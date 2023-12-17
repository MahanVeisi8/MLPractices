# In the following neural network that has a hidden layer, perform the backpropagation algorithm twice and update the weight vectors in the network with the gradient reduction method along with momentum. The initial values are given in the figure. Use the mean squared error function as the cost function. Sigmoid function functions.
# (the neural network has an input with size three (x1, x2, x3), a hidden layer with two neurons that is connected and to the output layer)
# input:
# x1 = 1, x2 = 0, x3 = 1
# weights of hidden layer:
# w11 = 0.2, w12 = -0.3, w21 = 0.4, w22 = 0.1, w31 = -0.5, w32 = -0.2, b1 = -0.4, b2 = 0.2
# output layer:
# u1 = -0.3, u2= -0.2, b3 = 0.1, 
# target:
# Y = 1
# learning rate = alpha = 0.9


import numpy as np

# sigmoid function
def 