"""
The main module of the program.
"""
import matrix
import power_method


def page_rank(input_filename="test.csv",
              input_names_filename="names_test.csv",
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
    adj_matrix, names = matrix.read_adj_matrix(input_filename,
                                               input_names_filename)
    matrix_G = matrix.matrix_G(
        matrix.matrix_S(adj_matrix),
        damping_factor, personalization_vector)
    pagerank_vector = power_method.power_method(matrix_G, max_iterations,
                                                epsilon)
    result_dict = [(list(names.values())[i], pagerank_vector[i]) for i
                   in range(len(pagerank_vector))]
    result_dict = sorted(result_dict, key=lambda t: t[1])[::-1]
    return result_dict


if __name__ == "__main__":
    print(page_rank())
