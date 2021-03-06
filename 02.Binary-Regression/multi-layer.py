import numpy as np


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def relu(input):
    return input * (input > 0)


def initialize_parameters(n_x, n_h, n_y, n_layer):
    # 편의를 위해서 seed 설정
    np.random.seed(20181001)
    parameters = dict()

    for n in range(n_layer):
        if n + 1 == 1:
            # W1.shape = (n_x, n_h)
            # b1.shape = (1, n_h)
            parameters["W1"] = np.random.randn(n_x, n_h)
            parameters["b1"] = np.zeros([1, n_h])
        elif n + 1 == n_layer:
            # Wn.shape = (n_h, n_h)
            # bn.shape = (1, n_h)
            parameters["W" + str(n + 1)] = np.random.randn(n_h, n_y)
            parameters["b" + str(n + 1)] = np.zeros([1, n_y])
        else:
            # WL.shape = (n_h, n_y)
            # bL.shape = (1, n_y)
            parameters["W" + str(n + 1)] = np.random.randn(n_h, n_h)
            parameters["b" + str(n + 1)] = np.zeros([1, n_h])
    return parameters


def forward_propagation(X, parameters, activation="sigmoid"):
    caches = dict()
    # Z1.shape = (m, n_h) = (m, n_x) * (n_x, n_h)
    # Zn.shape = (m, n_h) = (m, n_h) * (n_h, n_h)
    # ZL.shape = (m, n_y) = (m, n_h) * (n_h, n_y)
    n_layer = int(len(parameters) / 2)
    A = X
    for n in range(n_layer):
        Z = np.dot(A, parameters["W" + str(n + 1)]) + parameters["b" + str(n + 1)]
        A = sigmoid(Z)
        caches["A" + str(n + 1)] = A
    return A, caches


def compute_loss(Y_hat, Y):
    return -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / len(Y)


def backward_propagation(X, Y, caches):
    # dL/dWL = dL/dZL * dZL/dWL
    # dL/dbL = dL/dZL * dZL/dbL

    # dL/dWn = dL/dZL * dZL/dZL-1 * dZL-1/dZL-2 * ... * dZn/dWn
    # dL/dbn = dL/dZL * dZL/dZL-1 * dZL-1/dZL-2 * ... * dZn/dbn

    # by Chain rule

    grads = dict()
    n_layer = len(caches)
    for n in reversed(range(n_layer)):

        if n + 1 == n_layer:
            dL_dZ = (caches["A" + str(n + 1)] - Y)
        else:
            dL_dZ = np.dot(dL_dZ, parameters["W" + str(n + 2)].T) * caches["A" + str(n + 1)] * (
                        1 - caches["A" + str(n + 1)])

        if n == 0:
            dZ_dW = X
        else:
            dZ_dW = caches["A" + str(n)]

        dZ_db = 1
        dL_dW = np.dot(dL_dZ.T, dZ_dW)
        dL_db = np.sum(dL_dZ * dZ_db)

        grads["dW" + str(n + 1)] = dL_dW
        grads["db" + str(n + 1)] = dL_db
    return grads


def update_parameters(parameters, grads, learning_rate):
    n_layer = int(len(parameters) / 2)
    for n in range(n_layer):
        parameters["W" + str(n + 1)] -= learning_rate * grads["dW" + str(n + 1)].T
        parameters["b" + str(n + 1)] -= learning_rate * grads["db" + str(n + 1)]

    return parameters


# Data. X.shape = (4, 2), Y.shape = (4, 1)
X = np.array([[1, 2], [3, 4], [2, 1], [4, 3]])
Y = np.array([[0], [1], [0], [1]])
print(X)
print(Y)

# Hyperparamerters
num_epochs = 1000
learning_rate = 1e-1
num_layers = 4

# 1. Initialize Parameters
parameters = initialize_parameters(X.shape[1], 4, Y.shape[1], num_layers)

# 2. Loop N iteration (N: Num of epochs)
for epoch in range(num_epochs):
    # Forward Probagation
    Y_hat, caches = forward_propagation(X, parameters)
    # Compute loss
    loss = compute_loss(Y_hat, Y)

    # Backward Propagation
    grads = backward_propagation(X, Y, caches)

    # Update Parameters
    parameters = update_parameters(parameters, grads, learning_rate)

    # Print Loss
    if (epoch + 1) % 100 == 0 or epoch + 1 == 1:
        print(epoch + 1, loss)

print(Y)
print(Y_hat)


