import numpy as np


def count_new_pointers(pointers):
    # to calculate new pointers put in cell sum of self and all previous values
    for i in range(len(pointers) - 2, -1, -1):
        pointers[i] += pointers[i + 1]


def polynomial_features(data, degree=2):
    # new matrix with added combinations of variables
    if degree >= 2:
        # float64 is used not to exceed limits so easily
        # object type would not exceed but is slower than float
        if data.dtype is not np.dtype('float64'):
            data = data.astype('float64')
        # number of dimensions is the length of row
        dimensions = np.size(data, axis=1)
        # column where multiplying should start to make
        # next degree of the polynomial and use previous results
        end = dimensions
        # array of pointers showing where current dimension should start
        # multiplying not to repeat results - to avoid x1 * x2 and x2 * x1 to be considered as different
        # length of array - first cell shows where x1 (column) dimension should start
        # multiplying other columns until the end of array
        pointers = [i for i in range(dimensions, 0, -1)]
        # scans over degrees of polynomial
        for _ in range(1, degree):
            # scans over dimensions
            for dimension, pointer in enumerate(pointers):
                # multiplies result of previous outer loop by current dimension
                # so for example all results from previous will be multiplied by x1
                # note: dimension varies from 1 to number of dimensions so bias is not included
                # note: biased_data has to be reshaped to multiply properly
                tmp = data[:, end - pointer:end] * data[:, dimension].reshape(-1, 1)
                # concatenates horizontally result of multiplying to ending matrix
                data = np.concatenate((data, tmp), axis=1)
            end = np.size(data, axis=1)
            # calculate new pointers by summing all previous
            count_new_pointers(pointers)
    return data
