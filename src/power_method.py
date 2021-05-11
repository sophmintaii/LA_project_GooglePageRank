"""
Contains implementation of PageRank algorithm using
classical power method.
"""
import numpy as np


def power_method(matrix_G, max_iterations=1000, epsilon=0.00001):
    """
    Finds PageRank through the classical power method
    for given Google matrix G.
    :param epsilon: maximal error.
    :param max_iterations: maximal number of iterations.
    :param matrix_G: Google matrix.
    :return: PageRank vector.
    """
    n_pages = len(matrix_G)
    current_pr = np.array([1 / n_pages for _ in range(n_pages)])
    for iteration in range(max_iterations):
        prev_pr = current_pr.copy()
        current_pr = np.dot(current_pr, matrix_G)
        if np.linalg.norm(prev_pr - current_pr) < epsilon:
            return current_pr
    return current_pr
