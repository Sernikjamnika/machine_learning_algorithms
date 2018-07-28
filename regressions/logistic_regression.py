import numpy as np


def decision_boundary(probability):
    """
    Decides if sample belongs to first set or the second one
    :param probability: probability of sample being 1
    :return: predicted label
    """
    return 1 if probability >= 0.5 else 0


def classify(predictions):
    """
    Classifies all given predictions
    :param predictions: vector of percentages(1 x m)
    :return: vector of classes (1 x m)
    """
    return np.array([decision_boundary(prediction) for prediction in predictions])


def sigmoid_function(x, theta):
    """
    Calculates the result of sigmoid function
    :param x: features (m x n)
    :param theta: vector of weights (n x 1)
    :return:
    """
    return 1 / (1 + np.exp(-np.dot(x, theta)))


def logistic_regression(x, y, learning_rate=0.01, number_of_iter=1000, add_bias=False):
    """
    Function used on the learning set to calculate vector of weight
    :param x: features (m x n)
    :param y: labels (m x 1)
    :param learning_rate: tells how fast algorithm learns
    :param number_of_iter: number of iterations of gradient descent
    :param add_bias: tells if add bias
    :return: vector of weights (n x 1)
    """
    # NOTE: this algorithm is similar to Linear Regression
    # the only thing that is changed is the cost function
    if add_bias:
        x = np.c_[np.ones((len(x), 1)), x]
    # choosing random starting point
    theta = np.zeros(x.shape[1])
    x_transposed = x.T
    for _ in range(number_of_iter):
        prediction = sigmoid_function(x=x, theta=theta)
        gradient = np.dot(x_transposed, (y - prediction))
        theta += learning_rate * gradient
    return theta
