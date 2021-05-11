from _csv import reader

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score


def nxpg(names, edges):
    """
    calculating pagerank using built-in function from networkx
    """
    with open(names, "r") as filein:
        csv_reader = reader(filein)
        names = list(map(tuple, csv_reader))[1:]
    names = dict(names)
    nodes = [str(node) for node in range(0, len(names))]

    with open(edges, "r") as filein:
        csv_reader = reader(filein)
        edges = list(map(tuple, csv_reader))[1:]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for el in edges:
        G.add_edge(el[0], el[1])

    return nx.pagerank(G, alpha=0.85)


def MAP_comparison(pagerank, test):
    """
    Computing the mean average precision
    """
    y_true = []
    for i in range(len(test)):
        if pagerank[i][0] == test[i][0]:
            y_true.append(1)
        else:
            y_true.append(0)
    aps = average_precision_score(y_true, [int(x[1]) for x in pagerank])
    return aps


def normalised_kendall_tau_distance(values1, values2):
    """
    Computing the Kendall tau distance
    """
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]),
                                np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1))


def avg_distance(list1, list2: list, k=50):
    """
    Computing average distance between the same entry in two rankings
    """
    distances = []
    for ind in range(min(k, len(list1))):
        el = list1[ind]
        ind2 = list2.index(el)
        distance = abs(ind - ind2)
        distances.append(distance)
    return np.average(distances)


def prob_difference(list1, list2):
    """
    Computing average difference between the probabilities of same entry in two rankings
    """
    ss = []
    list_1 = sorted(list1, key=lambda x: x[0])
    list_2 = sorted(list2, key=lambda x: x[0])
    for i in range(len(list_1)):
        d = abs(list_1[i][1] - list_2[i][1])
        ss.append(d)
    return ss
