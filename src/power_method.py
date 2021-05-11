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

    result_lst = []
    result_lst.append((current_pr, "initial"))
    result_lst.append((current_pr, "initial"))

    for iteration in range(max_iterations):
        prev_pr = current_pr.copy()
        current_pr = np.dot(current_pr, matrix_G)
        result_lst.append((current_pr, iteration))
        if np.linalg.norm(prev_pr - current_pr) < epsilon:
            print(f"Number of iterations for classical pagerank = "
                  f"{iteration + 1}")
            return result_lst
    return result_lst


def adaptive_power_method(matrix_G, max_iterations=1000, epsilon=0.00001,
                          adapt_epsilon=0.001):
    """
    Finds PageRank through the adaptive power method
    for given Google matrix G.
    :param adapt_epsilon: maximal error of convergence for one entry.
    :param epsilon: maximal error.
    :param max_iterations: maximal number of iterations.
    :param matrix_G: Google matrix.
    :return: PageRank vector.
    """
    fixed = set()
    n_pages = len(matrix_G)
    current_pr = np.array([1 / n_pages for _ in range(n_pages)])

    result_lst = []
    result_lst.append((current_pr, "initial"))
    result_lst.append((current_pr, "initial"))

    for iteration in range(max_iterations):
        prev_pr = current_pr.copy()

        current_pr = vector_matrix_mult(current_pr, matrix_G, fixed)
        for i in range(len(current_pr)):
            if i not in fixed:
                if abs(current_pr[i] - prev_pr[i]) < adapt_epsilon:
                    fixed.add(i)

        result_lst.append((current_pr, iteration))
        if np.linalg.norm(prev_pr - current_pr) < epsilon:
            print(f"Number of iterations for adaptive power method pagerank "
                  f"= {iteration + 1}")
            return result_lst
    return result_lst


def vector_matrix_mult(vector, matrix, fixed):
    """
    Performs matrix-vector multiplication. For elements with specific indexes
    does not perform multiplication, but keeps them the same as the
    corresponding elements in initial vector.
    :param vector: vector to multiply
    :param matrix: matrix to multiply
    :param fixed: set of element indexes which won't be changed
    :return: resultant vector
    """
    result_vector = vector.copy()
    for i in range(len(vector)):
        if i not in fixed:
            result_vector[i] = np.dot(np.transpose(matrix)[i], vector)
    return result_vector
