"""
    This program draws our forest. The input file should have a dictionary
    in it that has this structure:

    d = {1: [(0,0), [2,3], 15],
         2: [(10,0), [1], 10],
         3: [(0,10), [1], 20]}

         id_number: [coordinates, list of neighbors, radius/age]
"""

import networkx as nx
import matplotlib.pyplot as plt
import sys
import pickle
from matplotlib.axes._axes import _log as matplotlib_axes_logger


def generate_graph():
    """ Returns a graph with desired characteristics. """

    positions = {0: (0, 0), 1: (1, 1), 2: (1, 2), 3: (2, 1), 4: (2, 2)}
    # G = nx.Graph(5, 20, pos = positions)
    G = nx.random_geometric_graph(5, 20, pos = positions)
    edges = [(0, 1), (1, 2)]
    return G, positions, edges

def main():
    """ The main function. """

    # This represses some weird warnings that I was getting:
    matplotlib_axes_logger.setLevel('ERROR')

    # Read the dictionary from the data file:
    with open('testdata.txt', 'rb') as fn:
        dct = pickle.loads(fn.read())

    # Create the graph:
    G = nx.Graph()

    # Get various data structures from the dictionary:
    coords = {key: value[0] for key, value in dct.items()}
    edges = [(k, e) for k, v in dct.items() for e in v[1] if e > k]
    radii = [value[2] for value in dct.values()]
    labels = {key: value[0] for key, value in dct.items()}

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    # Create the figure and axes, and configure the size of the output:
    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw edges and then nodes:

    nx.draw_networkx_edges(G,
                           coords,
                           width = 5,
                           alpha = 0.5,
                           style = 'dashed',
                           solid_capstyle = 'round'
                           )

    nx.draw_networkx_nodes(G,
                           coords,
                           node_size = radii,
                           node_color = 'g',
                           alpha = 0.9
                           )

    nx.draw_networkx_labels(G,
                            coords,
                            labels,
                            font_size = 24,
                            font_color = 'b',
                            alpha = 0.8)

    # plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
