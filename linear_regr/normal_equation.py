import numpy as np


def normal_equation(x, y):
    """
    Returns the vector of weights which fits the hyperplane best
    :param x: parameters of given data (m x n)
    :param y: labels (m x 1)
    :return: vector of weights - theta (n x 1)
    """
    # checking if given data is numpy ndarray
    if type(x) is not np.ndarray:
        x = np.asarray(x)
    if type(y) is not np.ndarray:
        y = np.asarray(y)
    # adding bias to given data
    x_biased = np.c_[np.ones((len(x), 1)), x]
    # transposing x_biased once to use in normal equation
    x_transposed = x_biased.T
    theta = np.linalg.inv(x_transposed.dot(x_biased)).dot(x_transposed.dot(y))
    return theta
