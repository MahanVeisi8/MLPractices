# -*- coding: utf-8 -*-
"""MLHW3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uw2xl7NoCt-AXFrcmO6bnhV8DhMtW49w
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

w11, w12, w21, w22, w31, w32 = 0.2, -0.3, 0.4, 0.1, -0.5, -0.2
b1, b2 = -0.4, 0.2
u1, u2 = -0.3, -0.2
b3 = 0.1

momentum_w11, momentum_w12, momentum_w21, momentum_w22, momentum_w31, momentum_w32 = 0, 0, 0, 0, 0, 0
momentum_b1, momentum_b2 = 0, 0
momentum_u1, momentum_u2, momentum_b3 = 0, 0, 0

x1, x2, x3 = 1, 0, 1

target = 1

alpha = 0.9
beta = 0.8

epochs = 2

print("Initial Values:")
print("w11:", w11, "w12:", w12, "w21:", w21, "w22:", w22, "w31:", w31, "w32:", w32)
print("b1:", b1, "b2:", b2)
print("u1:", u1, "u2:", u2)
print("b3:", b3)
print("-------------------------")

for epoch in range(epochs):
    z1 = w11 * x1 + w21 * x2 + w31 * x3 + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = w12 * x1 + w22 * x2 + w32 * x3 + b2
    a2 = 1 / (1 + np.exp(-z2))

    z3 = u1 * a1 + u2 * a2 + b3
    a3 = 1 / (1 + np.exp(-z3))

    loss = 0.5 * (a3 - target) ** 2

    delta_z3 = (a3 - target) * a3 * (1 - a3)

    delta_z1 = (u1 * delta_z3) * a1 * (1 - a1)
    delta_z2 = (u2 * delta_z3) * a2 * (1 - a2)

    momentum_w11 = beta * momentum_w11 + (1 - beta) * delta_z1 * x1
    momentum_w12 = beta * momentum_w12 + (1 - beta) * delta_z2 * x1
    momentum_w21 = beta * momentum_w21 + (1 - beta) * delta_z1 * x2
    momentum_w22 = beta * momentum_w22 + (1 - beta) * delta_z2 * x2
    momentum_w31 = beta * momentum_w31 + (1 - beta) * delta_z1 * x3
    momentum_w32 = beta * momentum_w32 + (1 - beta) * delta_z2 * x3

    momentum_b1 = beta * momentum_b1 + (1 - beta) * delta_z1
    momentum_b2 = beta * momentum_b2 + (1 - beta) * delta_z2

    momentum_u1 = beta * momentum_u1 + (1 - beta) * delta_z3 * a1
    momentum_u2 = beta * momentum_u2 + (1 - beta) * delta_z3 * a2
    momentum_b3 = beta * momentum_b3 + (1 - beta) * delta_z3

    w11 -= alpha * momentum_w11
    w12 -= alpha * momentum_w12
    w21 -= alpha * momentum_w21
    w22 -= alpha * momentum_w22
    w31 -= alpha * momentum_w31
    w32 -= alpha * momentum_w32

    b1 -= alpha * momentum_b1
    b2 -= alpha * momentum_b2

    u1 -= alpha * momentum_u1
    u2 -= alpha * momentum_u2
    b3 -= alpha * momentum_b3

    if epoch % 1 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.6f}")
        print("w11:", w11, "| w12:", w12, "| w21:", w21, "| w22:", w22, "| w31:", w31, "| w32:", w32)
        print("b1:", b1, "| b2:", b2)
        print("u1:", u1, "| u2:", u2)
        print("b3:", b3)
        print("momentum_w11:", momentum_w11, "| momentum_w12:", momentum_w12)
        print("momentum_w21:", momentum_w21, "| momentum_w22:", momentum_w22)
        print("momentum_w31:", momentum_w31, "| momentum_w32:", momentum_w32)
        print("momentum_b1:", momentum_b1, "| momentum_b2:", momentum_b2)
        print("momentum_u1:", momentum_u1, "| momentum_u2:", momentum_u2)
        print("momentum_b3:", momentum_b3)
        print("-------------------------")

print("Final Values:")
print("w11:", w11, "| w12:", w12, "| w21:", w21, "| w22:", w22, "| w31:", w31, "| w32:", w32)
print("b1:", b1, "|b2:", b2)
print("u1:", u1, "|u2:", u2)
print("b3:", b3)
print("momentum_w11:", momentum_w11, "| momentum_w12:", momentum_w12)
print("momentum_w21:", momentum_w21, "| momentum_w22:", momentum_w22)
print("momentum_w31:", momentum_w31, "| momentum_w32:", momentum_w32)
print("momentum_b1:", momentum_b1, "| momentum_b2:", momentum_b2)
print("momentum_u1:", momentum_u1, "| momentum_u2:", momentum_u2)
print("momentum_b3:", momentum_b3)
print("-------------------------")

prediction = sigmoid(u1 * sigmoid(w11 * x1 + w21 * x2 + w31 * x3 + b1) +
                     u2 * sigmoid(w12 * x1 + w22 * x2 + w32 * x3 + b2) + b3)

print("Prediction:", prediction)
print("Final Loss:", loss)



import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

w11, w12, w21, w22, w31, w32 = 0.2, -0.3, 0.4, 0.1, -0.5, -0.2
b1, b2 = -0.4, 0.2
u1, u2 = -0.3, -0.2
b3 = 0.1

momentum_w11, momentum_w12, momentum_w21, momentum_w22, momentum_w31, momentum_w32 = 0, 0, 0, 0, 0, 0
momentum_b1, momentum_b2 = 0, 0
momentum_u1, momentum_u2, momentum_b3 = 0, 0, 0

x1, x2, x3 = 1, 0, 1

target = 1

alpha = 0.9
beta = 0.8

epochs = 1000

print("Initial Values:")
print("w11:", w11, "w12:", w12, "w21:", w21, "w22:", w22, "w31:", w31, "w32:", w32)
print("b1:", b1, "b2:", b2)
print("u1:", u1, "u2:", u2)
print("b3:", b3)
print("-------------------------")

for epoch in range(epochs):
    z1 = w11 * x1 + w21 * x2 + w31 * x3 + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = w12 * x1 + w22 * x2 + w32 * x3 + b2
    a2 = 1 / (1 + np.exp(-z2))

    z3 = u1 * a1 + u2 * a2 + b3
    a3 = 1 / (1 + np.exp(-z3))

    loss = 0.5 * (a3 - target) ** 2

    delta_z3 = (a3 - target) * a3 * (1 - a3)

    delta_z1 = (u1 * delta_z3) * a1 * (1 - a1)
    delta_z2 = (u2 * delta_z3) * a2 * (1 - a2)

    momentum_w11 = beta * momentum_w11 + (1 - beta) * delta_z1 * x1
    momentum_w12 = beta * momentum_w12 + (1 - beta) * delta_z2 * x1
    momentum_w21 = beta * momentum_w21 + (1 - beta) * delta_z1 * x2
    momentum_w22 = beta * momentum_w22 + (1 - beta) * delta_z2 * x2
    momentum_w31 = beta * momentum_w31 + (1 - beta) * delta_z1 * x3
    momentum_w32 = beta * momentum_w32 + (1 - beta) * delta_z2 * x3

    momentum_b1 = beta * momentum_b1 + (1 - beta) * delta_z1
    momentum_b2 = beta * momentum_b2 + (1 - beta) * delta_z2

    momentum_u1 = beta * momentum_u1 + (1 - beta) * delta_z3 * a1
    momentum_u2 = beta * momentum_u2 + (1 - beta) * delta_z3 * a2
    momentum_b3 = beta * momentum_b3 + (1 - beta) * delta_z3

    w11 -= alpha * momentum_w11
    w12 -= alpha * momentum_w12
    w21 -= alpha * momentum_w21
    w22 -= alpha * momentum_w22
    w31 -= alpha * momentum_w31
    w32 -= alpha * momentum_w32

    b1 -= alpha * momentum_b1
    b2 -= alpha * momentum_b2

    u1 -= alpha * momentum_u1
    u2 -= alpha * momentum_u2
    b3 -= alpha * momentum_b3

    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.6f}")
        print("w11:", w11, "| w12:", w12, "| w21:", w21, "| w22:", w22, "| w31:", w31, "| w32:", w32)
        print("b1:", b1, "| b2:", b2)
        print("u1:", u1, "| u2:", u2)
        print("b3:", b3)
        print("momentum_w11:", momentum_w11, "| momentum_w12:", momentum_w12)
        print("momentum_w21:", momentum_w21, "| momentum_w22:", momentum_w22)
        print("momentum_w31:", momentum_w31, "| momentum_w32:", momentum_w32)
        print("momentum_b1:", momentum_b1, "| momentum_b2:", momentum_b2)
        print("momentum_u1:", momentum_u1, "| momentum_u2:", momentum_u2)
        print("momentum_b3:", momentum_b3)
        print("-------------------------")

print("Final Values:")
print("w11:", w11, "| w12:", w12, "| w21:", w21, "| w22:", w22, "| w31:", w31, "| w32:", w32)
print("b1:", b1, "|b2:", b2)
print("u1:", u1, "|u2:", u2)
print("b3:", b3)
print("momentum_w11:", momentum_w11, "| momentum_w12:", momentum_w12)
print("momentum_w21:", momentum_w21, "| momentum_w22:", momentum_w22)
print("momentum_w31:", momentum_w31, "| momentum_w32:", momentum_w32)
print("momentum_b1:", momentum_b1, "| momentum_b2:", momentum_b2)
print("momentum_u1:", momentum_u1, "| momentum_u2:", momentum_u2)
print("momentum_b3:", momentum_b3)
print("-------------------------")

prediction = sigmoid(u1 * sigmoid(w11 * x1 + w21 * x2 + w31 * x3 + b1) +
                     u2 * sigmoid(w12 * x1 + w22 * x2 + w32 * x3 + b2) + b3)

print("Prediction:", prediction)
print("Final Loss:", loss)

