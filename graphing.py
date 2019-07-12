"""
    This program draws our forest. The input file should have a dictionary
    in it that has this structure:

    d = {1: [(0,0), [2,3], 15],
         2: [(10,0), [1], 10],
         3: [(0,10), [1], 20]}

         id_number: [coordinates, list of neighbors, radius/age]

    d = {(0,0): [(1,1), (2,3)],
         (1,1): [(0,0)],
         (2,3): [(0,0)]
         }

    locations = [<tree object>, <tree object>, <tree object>]



"""

import networkx as nx
import matplotlib.pyplot as plt
import math
import os


def graph(trees, name):
    """ The main function. """

    # Create the graph:
    G = nx.Graph()

    coords = {trees[i]: trees[i].location
                               for i in range(len(trees))
                               if trees[i]}

    edges = [(t, e) for t in coords for e in t.neighbors]

    labels = {t: t.__repr__() for t in coords}

    for edge in edges:
        G.add_edge(edge[0], edge[1])

    # Create the figure and axes, and configure the size of the output:
    fig, ax = plt.subplots(figsize=(5, 5))

    # Draw edges and then nodes:
    nx.draw_networkx_edges(G,
                           coords,
                           width = 1,
                           alpha = 0.5,
                           solid_capstyle = 'round'
                           )

    nx.draw_networkx_nodes(G,
                           coords,
                           node_size = 100,
                           node_color = 'yellow',
                           alpha = 0.9
                           )

    nx.draw_networkx_labels(G,
                            coords,
                            labels,
                            font_size = 8,
                            font_color = 'b',
                            alpha = 0.8)

    plt.savefig('graph' + str(name) + '.png')
    plt.close('all')


if __name__ == '__main__':
    main()
