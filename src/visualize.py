"""
The module contains visualization of PageRank calulation.
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.animation import FuncAnimation
import power_method
import matrix
import scipy

damping_factor = 0.85
personalization_vector = None
max_iterations = 1000
epsilon = 0.0001
input_filename = "../resourses/art.csv"
input_names_filename = "../resourses/art_names.csv"

adj_matrix, names, g = matrix.read_adj_matrix(input_filename,
                                              input_names_filename)
pos = nx.kamada_kawai_layout(g)
matrix_G = matrix.matrix_G(
    matrix.matrix_S(adj_matrix),
    damping_factor, personalization_vector)
rank_res = power_method.adaptive_power_method(matrix_G, max_iterations,
                                     epsilon)

# print(rank_res)

def s(l):
    while len(l) > 1:
        x = l.pop(0)
        yield x
    return l[0]


def v(i):
    rank, iteration = i
    rank = np.asarray(rank).ravel()
    g_nodes = g.nodes()
    plot_nodes = nx.draw_networkx_nodes(
        G=g,
        pos=pos,
        ax=ax,
        nodelist=g_nodes,
        node_color=rank,
        alpha=1,
        node_size=800,
        cmap=plt.cm.Purples,
        vmin=0,
        vmax=0.2
    )
    ax.axis("off")
    ax.set_title(f"Step {iteration}")
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, font_size=10)
    return [plot_nodes, ]


f, ax = plt.subplots()
ani = FuncAnimation(
    f,
    v,
    frames=s(rank_res),
    interval=500,
    blit=True
)
f.suptitle(f"  Page Rank")
ani.save("../ranking_progress_adaptive.gif", writer='imagemagick')
