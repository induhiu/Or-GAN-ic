"""
    I'd like to create a simple forest where I can:
    -- specify the location of the trees on a lattice
    -- color the nodes based on a tree's health
    -- have the size of each node indicate the tree's age
    -- draw edges between neighboring trees
"""

import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


def generate_graph():
    """ Returns a graph with desired characteristics. """

    positions = {0: (0, 0), 1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2)}
    # G = nx.Graph(5, 20, pos = positions)
    G = nx.random_geometric_graph(5, 20, pos = positions)
    edges = [(0, 1), (1, 2)]
    return G, positions, edges

def main():
    """ The main function. """

    G, positions, edges = generate_graph()
    fig, ax = plt.subplots(figsize=(10,7))
    nx.draw_networkx_nodes(G, positions, node_size=100)
    nx.draw_networkx_edges(G, positions, edges)
    # nx.draw(G)
    plt.show()


if __name__ == '__main__':
    main()
