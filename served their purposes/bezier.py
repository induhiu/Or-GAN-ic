"""
DATE: June 4, 2019
FILE: bezier.py
AUTH: Dave Perkins

DESC: This program helps generate a set of characters for the forest's
      'common language'. The squiggles are meant to look like chemical
      signatures. We want to use the program to create ten characters
      and 7000 similar copies of each, to mimic the MNIST database.
"""

import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
from sys import exit
from PIL import Image

# Our created images will be MAX by MAX pixels in size:
MAX = 28

# Once the user picks a character, we generate this many similar characters:
NUM_CHARACTERS_IN_FAMILY = 20

# The Bezier curves (the squiggles) require points, and I want to space them
# out a bit, so I'm defining 'quadrants' from which we will choose these
# points. To get more spacing, the quadrants are really the four corners
# of the image's square.

# Make this larger to increase the sizes of these corner squares:
width = 5

# Quadrants are measured from lower left to upper right:
Q0 = [] # for index fixing -- just ignore!
Q1 = [(MAX - width, MAX - width), (MAX, MAX)]
Q2 = [(0, MAX - width), (width, width)]
Q3 = [(0, 0), (width, width)]
Q4 = [(MAX - width, 0), (MAX, width)]

QUADRANTS = [Q0, Q1, Q2, Q3, Q4]


def bernstein_poly(i, n, t):
    """ Returns the Bernstein polynomial of n, i as a function of t. """

    return comb(n, i) * ( t ** (n - i) ) * (1 - t) ** i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ... [Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t)
                                 for i in range(0, nPoints)
                                ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def pick_points(quads):
    """
        Returns a list of lists of tuples.

        quads is a list of quadrants, like [1, 2, 4, 3, 1], from which we
        will choose our random points (in the given order).

        Like this:

        [ [(2, 4), (7, 2), (3, 4), (1, 9)] ,
          [(3, 3), (8, 2), (5, 6), (4, 4)] ,
          [(9, 2), (7, 8), (5, 3), (3, 7)]
        ]
    """

    return [(get_point_from_quadrant(QUADRANTS[quad])) for quad in quads]


def get_point_from_quadrant(quad):
    """ Returns a point from the given quadrant. """

    # Get a random tuple from the range (0, 1) in each direction:
    point = np.random.rand(1, 2)

    # Transform that tuple so it fits inside the given quadrant:
    x = quad[0][0] + 4 * point[0, 0]
    y = quad[0][1] + 4 * point[0, 1]

    return x, y


def define_lines(num_lines):
    """ Returns num_lines lists of quadrants from which to choose points. """

    return [np.random.randint(1, 5, np.random.randint(4, 8))
            for _ in range(num_lines)
           ]


def create_character(fname, quad_list):
    """ Creates and saves a character with filename fname. """

    # Set the filename and path:
    fname = './characters/' + fname + '.png'

    # Unpack the figure and axis, and set figure to 1 inch by 1 inch:
    fig, ax = plt.subplots(figsize = (1, 1))

    # Turn the axes off so that we only see and save the image:
    ax.set_axis_off()

    # This list of lists stores all of the points, chosen from the various
    # quadrants, for each of the bezier curves we wish to draw:
    points_list = [pick_points(q) for q in quad_list]

    # Graph all of the curves:
    for lst in points_list:

        # Calculate a list of points to connect for each curve:
        xvals, yvals = bezier_curve(lst, nTimes=200)

        # Plot; use solid_capstyle for rounding line endpoints:
        plt.plot(xvals, yvals, 'k', linewidth = 2, solid_capstyle = 'round')

    # Set the plot's axes to match the desired image dimensions:
    plt.xlim(0, MAX)
    plt.ylim(0, MAX)

    # Save the figure (in color):
    plt.savefig(fname, dpi = MAX)
    plt.close()

    # Convert to grayscale:
    Image.open(fname).convert('L').save(fname)


def main():
    """ The main function. """

    # A flag that allows the user to view multiple characters before picking:
    filename = ''

    while filename == '':

        # Pick a random number of lines to use, and generate quadrants for each:
        quad_list = define_lines(np.random.randint(2, 4))

        # Make one character to see if it's a keeper:
        create_character('test', quad_list)

        # If the user hits Enter, we continue generating test characters:
        filename = input('Enter a filename to keep this character:')

    # Create the appropriate number of similar characters to the user's choice:
    for pic in range(NUM_CHARACTERS_IN_FAMILY):
        create_character('dave_images/' + filename + str(pic), quad_list)


if __name__ == "__main__":
    main()
