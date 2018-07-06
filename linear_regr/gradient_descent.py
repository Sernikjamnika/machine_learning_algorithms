import numpy as np


def gradient_descent(x, y, learning_rate=0.01, number_of_iter=1000):
    """
    Function using gradient descent method to
    compute best fitting hyperplane
    :param x: parameters of data given (m x n)
    :param y: labels (m x 1)
    :param learning_rate: parameter telling how fast algorithm learns
    :param number_of_iter: number of iterations of gradient descent
    :return: theta - vector of weights fitting hyperplane best (n x 1)
    """
    if type(x) is not np.ndarray:
        x = np.asarray(x)
    if type(y) is not np.ndarray:
        y = np.asarray(y)
        # adding bias to given data
    x_biased = np.c_[np.ones((len(x), 1)), x]
    # choosing random starting point
    theta = np.random.randn(2, 1)
    x_biased_transposed = x_biased.T
    constant = 1/len(x)
    for _ in range(number_of_iter):
        tmp_theta = constant * x_biased_transposed.dot(x_biased.dot(theta) - y)
        theta = theta - learning_rate * tmp_theta
    return theta
