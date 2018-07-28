import numpy as np


def normal_equation(x, y, add_bias=False):
    """
    Returns the vector of weights which fits the hyperplane best
    :param x: features (n x m)
    :param y: labels (m x 1)
    :param add_bias: tells if add bias
    :return: vector of weights - theta (n x 1)
    """
    # adding bias to given data
    if add_bias:
        x = np.c_[np.ones((len(x), 1)), x]
    # transposing x_biased once to use in normal equation
    x_transposed = x.T
    theta = np.linalg.inv(x_transposed.dot(x)).dot(x_transposed.dot(y))
    return theta
