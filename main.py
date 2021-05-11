"""
The main module of the program.
"""
from src import matrix, power_method
import time


def page_rank(input_filename="resourses/art.csv",
              input_names_filename="resourses/art_names.csv",
              damping_factor=0.85,
              personalization_vector=None,
              max_iterations=1000,
              epsilon=0.0001):
    """
    Calculates PageRank vector for data in given file.
    :param input_names_filename: csv files with id and nsme of the page.
    :param input_filename: csv file with adjacency list.
    :param damping_factor: P(hyperlinking).
    :param personalization_vector: vector of probabilities.
    :return: PageRank vector.
    """
    adj_matrix, names, g = matrix.read_adj_matrix(input_filename,
                                               input_names_filename)
    matrix_G = matrix.matrix_G(
        matrix.matrix_S(adj_matrix),
        damping_factor, personalization_vector)

    start = time.time()
    pagerank_result = power_method.power_method(matrix_G, max_iterations,
                                                epsilon)
    end = time.time()
    time_1 = end-start
    pagerank_vector = pagerank_result[-1][0]
    print("Computational time for classic pagerank = ", time_1)

    start = time.time()
    pagerank_result_adaptive = power_method.adaptive_power_method(matrix_G,
                                                                  max_iterations, epsilon)
    end = time.time()
    time_2 = end - start
    pagerank_vector_adaptive = pagerank_result_adaptive[-1][0]
    print("Computational time for pagerank with adaptive method = ", time_2)

    print(len(list(names.values())), len(pagerank_vector))
    result_dict = [(list(names.values())[i], pagerank_vector[i]) for i
                   in range(len(pagerank_vector))]
    result_dict = sorted(result_dict, key=lambda t: t[1])[::-1]

    result_dict_adaptive = [(list(names.values())[i], pagerank_vector_adaptive[i]) for i
                            in range(len(pagerank_vector_adaptive))]
    result_dict_adaptive = sorted(result_dict_adaptive, key=lambda t: t[1])[::-1]
    return result_dict, result_dict_adaptive


if __name__ == "__main__":
    result1, result2 = page_rank()
    for i in range(len(result1)):
        print(i+1, result1[i], result2[i])
