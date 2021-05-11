"""
The main module of the program.
"""
from sys import argv

from src import matrix, power_method, metrics,visualize
import time
import numpy as np


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

    result_dict = [(list(names.values())[i], pagerank_vector[i]) for i
                   in range(len(pagerank_vector))]
    result_dict = sorted(result_dict, key=lambda t: t[1])[::-1]

    result_dict_adaptive = [(list(names.values())[i], pagerank_vector_adaptive[i]) for i
                            in range(len(pagerank_vector_adaptive))]
    result_dict_adaptive = sorted(result_dict_adaptive, key=lambda t: t[1])[::-1]

    netxpage = np.array(list(metrics.nxpg(input_names_filename,input_filename).items()))
    result_netx = sorted(netxpage, key=lambda t: t[1])[::-1]
    for ind in range(len(result_netx)):
        new_el = (names[result_netx[ind][0]], float(result_netx[ind][1]))
        result_netx[ind] = new_el

    return result_dict, result_dict_adaptive, result_netx


if __name__ == "__main__":
    in_edges, in_names = argv[1:]
    visualize.visualition(in_edges, in_names)
    result1, result2, netx = page_rank(in_edges, in_names)
    for i in range(len(result1)):
        print(i+1, result1[i], result2[i], netx[i])
    print("-------------- POWER AND ADAPTIVE --------------")
    print('MAP comparison: ',metrics.MAP_comparison(result1,result2))
    print('Kendall Tau: ', metrics.normalised_kendall_tau_distance([x[0] for x in result1],[x[0] for x in result2]))
    print('AVG misplacement: ', metrics.avg_distance([x[0] for x in result1],[x[0] for x in result2]))
    print('AVG probability diff: ',np.average(metrics.prob_difference(result1,result2)))

    print("-------------- POWER AND NETWORKX --------------")
    print('MAP comparison: ', metrics.MAP_comparison(result1, netx))
    print('Kendall Tau: ', metrics.normalised_kendall_tau_distance([x[0] for x in result1], [x[0] for x in netx]))
    print('AVG misplacement: ', metrics.avg_distance([x[0] for x in result1], [x[0] for x in netx]))
    print('AVG probability diff: ', np.average(metrics.prob_difference(result1, netx)))

    print("-------------- ADAPTIVE AND NETWORKX--------------")
    print('MAP comparison: ', metrics.MAP_comparison(netx, result2))
    print('Kendall Tau: ', metrics.normalised_kendall_tau_distance([x[0] for x in netx], [x[0] for x in result2]))
    print('AVG misplacement: ', metrics.avg_distance([x[0] for x in netx], [x[0] for x in result2]))
    print('AVG probability diff: ', np.average(metrics.prob_difference(netx, result2)))