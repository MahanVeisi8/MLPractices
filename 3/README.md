# README - Practice Number 3: Simple Neural Network Implementation and Training

## Introduction

Welcome to Practice Number 3! In this practice, we will implement a simple neural network with a hidden layer and train it using the backpropagation algorithm. The neural network will use sigmoid activation functions and the error squared criterion function as the cost. We'll update the weights using the reduction plus momentum method.

## Implementation Details

We'll implement the neural network in Python using NumPy for numerical computations. The network will consist of an input layer, a hidden layer, and an output layer.

### Sigmoid Activation Function

We define a sigmoid activation function and its derivative:

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

### Neural Network Parameters
We initialize the weights and biases of the neural network along with momentum terms:

```python
w11, w12, w21, w22, w31, w32 = 0.2, -0.3, 0.4, 0.1, -0.5, -0.2
b1, b2 = -0.4, 0.2
u1, u2 = -0.3, -0.2
b3 = 0.1

momentum_w11, momentum_w12, momentum_w21, momentum_w22, momentum_w31, momentum_w32 = 0, 0, 0, 0, 0, 0
momentum_b1, momentum_b2 = 0, 0
momentum_u1, momentum_u2, momentum_b3 = 0, 0, 0
```

### Training Parameters
We set the learning rate, momentum factor, and number of epochs:
```python
alpha = 0.9
beta = 0.8
epochs = 1000
```

## Training the Neural Network

We train the neural network for a specified number of epochs using backpropagation. The training process involves forward propagation to compute the output and loss, followed by backward propagation to update the weights and biases.
```python
for epoch in range(epochs):
    # Forward propagation
    ...

    # Backward propagation
    ...

    # Update weights and biases with momentum
    ...

    # Print progress
    ...

# Print final weights, biases, and loss
...

# Print final prediction
...
```

## Conclusion
In this practice, we implemented a simple neural network from scratch and trained it using the backpropagation algorithm. We updated the weights using the reduction plus momentum method and monitored the training progress. This exercise provides valuable insight into the inner workings of neural networks and their training process.
