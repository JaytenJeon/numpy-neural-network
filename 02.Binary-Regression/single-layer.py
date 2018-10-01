import numpy as np


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def relu(input):
    return input * (input > 0)


def initialize_parameters(n_x, n_y):
    # 편의를 위해서 seed 설정
    np.random.seed(20181001)

    # W.shape = (n_x, n_y)
    # b.shape = (1, n_y)

    W = np.random.randn(n_x, n_y)
    b = np.zeros([1, n_y])

    # 편의를 위해 dictionary 사용
    parameters = {"W": W,
                  "b": b}

    return parameters


def forward_propagation(X, parameters, activation="sigmoid"):
    # Z.shape = Y.shape = (m, n_y) = (m, n_x) * (n_x, n_y)
    Z = np.dot(X, parameters["W"]) + parameters["b"]
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z)

    return A


def compute_loss(Y_hat, Y):
    # MSE
    return -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / len(Y)


def backward_propagation(X, Y, A):
    # dL/dA = dL/dA * dA/dZ * dZ/dW
    # dL/db = dL/dA * dA/dZ * dZ/db
    # by Chain rule

    dL_dZ = A - Y

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
X = np.array([[1, 2], [3, 4], [2, 1], [4, 3]])
Y = np.array([[0], [1], [0], [1]])
print(X)
print(Y)

# Hyperparamerters
num_epochs = 2000
learning_rate = 1e-2

# 1. Initialize Parameters
parameters = initialize_parameters(X.shape[1], Y.shape[1])

# 2. Loop N iteration (N: Num of epochs)
for epoch in range(num_epochs):
    # Forward Probagation
    Y_hat = forward_propagation(X, parameters, "sigmoid")

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

