"""
    Graphs a 2D plot with random number of nodes and edges.
    The sizes and colors of the nodes vary.
    Positions are chosen at random, but could be specified.
"""

import networkx as nx
import random
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import _log as matplotlib_axes_logger

def generate_random_graph(n_nodes, radius, seed=None):

    if seed is not None:
        random.seed(seed)

    # Generate a dict of positions
    pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(n_nodes)}

    # Create random network
    G = nx.random_geometric_graph(n_nodes, radius, pos=pos)

    return G


def network_plot(G, angle, save=False):

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # Get the maximum number of edges adjacent to a single node
    edge_max = max([G.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(G.degree(i)/edge_max) for i in range(n)]

    # network plot
    with plt.style.context(('ggplot')):

        fig, ax = plt.subplots(figsize=(10,7))
        # ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]

            # Scatter plot
            ax.scatter(xi, yi, c=colors[key], s=20+20*G.degree(key), edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):

            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))

        # Plot the connecting lines
            ax.plot(x, y, c='black', alpha=0.5)

    # Hide the axes
    ax.set_axis_off()

    if save is not False:
        plt.savefig(str(angle).zfill(3)+".png")
        plt.close('all')
    else:
        plt.show()

    return

def main():

    # This represses some weird warnings that I was getting:
    matplotlib_axes_logger.setLevel('ERROR')

    n = 30
    G = generate_random_graph(n_nodes=n, radius=0.25, seed=1)
    network_plot(G,0, save=False)

if __name__ == '__main__':
    main()
