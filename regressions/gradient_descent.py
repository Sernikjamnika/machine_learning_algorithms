import numpy as np


def gradient_descent(x, y, learning_rate=0.01, number_of_iter=1000, add_bias=False):
    """
    Uses gradient descent method to
    compute best fitting hyperplane
    :param x: features (m x n)
    :param y: labels (m x 1)
    :param learning_rate: tells how fast algorithm learns
    :param number_of_iter: number of iterations of gradient descent
    :param add_bias: tells if add bias
    :return: vector of weights - theta (n x 1)
    """
    # adding bias to given data
    if add_bias:
        x = np.c_[np.ones((len(x), 1)), x]
    # initializing vector of weights
    theta = np.zeros((x.shape[1], 1))
    x_biased_transposed = x.T
    constant = 1 / len(x)
    for _ in range(number_of_iter):
        tmp_theta = constant * x_biased_transposed.dot(x.dot(theta) - y)
        theta -= learning_rate * tmp_theta
    return theta
