"""
This module contains implementations of the methods that allow
to work with matrices needed for PageRank algorithm.
"""
import networkx as nx
import numpy as np
from csv import reader
import pandas as pd


def read_adj_matrix(filename, filename_names):
    """
    Reads data from file to the np array with adjacency matrix.
    :param filename_names: csv file with node_id and node_name.
    :param filename: file to read.
    :return: adjacency list.
    """
    names = []
    with open(filename_names, "r") as filein:
        csv_reader = reader(filein)
        names = list(map(tuple, csv_reader))[1:]
    names = dict(names)
    nodes = [str(node) for node in range(0, len(names))]
    edges = []
    with open(filename, "r") as filein:
        csv_reader = reader(filein)
        edges = list(map(tuple, csv_reader))[1:]
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return nx.to_pandas_adjacency(g).to_numpy(), names, g


def matrix_S(adj_matrix):
    """
    Get stochastic matrix S from adjacency matrix of hyperlinked pages.
    :param adj_matrix: hyperlink matrix.
    :return: stochastic matrix S.
    """
    zero_rows = np.where(~adj_matrix.any(axis=1))[0]
    for row in zero_rows:
        adj_matrix[row] = [entry + 1 for entry in adj_matrix[row]]
    matrix_S = []
    for row in adj_matrix:
        links = sum(row)
        matrix_S.append([entry / links for entry in row])
    return np.array(matrix_S)


def matrix_G(matrix_S, damping_factor=0.85, personalization_vector=None):
    """
    Creates Google matrix G.
    :param matrix_S: stochastic matrix.
    :param damping_factor: P(hyperlinking)
    :param personalization_vector: vector of probabilities.
    :return: matrix G.
    """
    len_S = len(matrix_S)
    if personalization_vector is None:
        personalization_vector = np.array([[1 / len_S for _ in range(len_S)]])
    ones = np.array([[1] for _ in range(len(matrix_S))])
    matrix_G = matrix_S * damping_factor + \
               np.dot(ones, personalization_vector) * (1 - damping_factor)
    return matrix_G
