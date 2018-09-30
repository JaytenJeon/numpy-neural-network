import numpy as np


def initialize_parameters(n_x, n_y):
    # 편의를 위해서 seed 설정
    np.random.seed(20180929)

    # W.shape = (n_x, n_y)
    # b.shape = (1, n_y)

    W = np.random.randn(n_x, n_y)
    b = np.zeros([1, n_y])

    # 편의를 위해 dictionary 사용
    parameters = {"W": W,
                  "b": b}

    return parameters


def forward_propagation(X, parameters):
    # Z.shape = Y.shape = (m, n_y) = (m, n_x) * (n_x, n_y)
    Z = np.dot(X, parameters["W"]) + parameters["b"]

    return Z


def compute_loss(Y_hat, Y):
    # MSE
    return np.sum(np.square(Y_hat - Y)) / len(Y)


def backward_propagation(X, Y, Z):
    # dL/dW = dL/dZ * dZ/dW
    # dL/db = dL/dZ * dZ/db
    # by Chain rule

    dL_dZ = 2 * (Z - Y)
    dZ_dW = X
    dZ_db = 1
    dL_dW = np.dot(dL_dZ.T, dZ_dW)
    dL_db = np.sum(dL_dZ * dZ_db)

    grads = {"dW": dL_dW,
             "db": dL_db}

    return grads


def update_parameters(parameters, grads, learning_rate):
    parameters["W"] -= learning_rate * grads["dW"].T
    parameters["b"] -= learning_rate * grads["db"]

    return parameters


# Data. X.shape = (4, 3), Y.shape = (4, 1)
X = np.array([[100, 90, 95], [85, 75, 75], [100, 100, 100], [50, 40, 45]])
Y = np.array([[95], [80], [100], [50]])
print(X)
print(Y)

# Hyperparamerters
num_epochs = 1000
learning_rate = 1e-5

# 1. Initialize Parameters
parameters = initialize_parameters(X.shape[1], Y.shape[1])

# 2. Loop N iteration (N: Num of epochs)
for epoch in range(num_epochs):
    # Forward Probagation
    Y_hat = forward_propagation(X, parameters)

    # Compute loss
    loss = compute_loss(Y_hat, Y)

    # Backward Propagation
    grads = backward_propagation(X, Y, Y_hat)

    # Update Parameters
    parameters = update_parameters(parameters, grads, learning_rate)

    # Print Loss
    if (epoch + 1) % 100 == 0 or epoch + 1 == 1:
        print(epoch + 1, loss)

print(Y)
print(Y_hat)

