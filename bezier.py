import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
from sys import exit
from PIL import Image

# GLOBAL CONSTANTS:
MY_DPI = 255
MAX_COORD = 28
NUM_CURVES = 3
NUM_POINTS_IN_CURVE = 4

# Quadrants are measured from lower left to upper right:
Q0 = []
# Q1 = [(MAX_COORD // 2, MAX_COORD // 2), (MAX_COORD, MAX_COORD)]
# Q2 = [(0, MAX_COORD // 2), (MAX_COORD // 2, MAX_COORD)]
# Q3 = [(0, 0), (MAX_COORD // 2, MAX_COORD // 2)]
# Q4 = [(MAX_COORD // 2, 0), (MAX_COORD, MAX_COORD // 2)]
Q1 = [(18, 18), (22, 22)]
Q2 = [(1, 18), (5, 22)]
Q3 = [(1, 1), (5, 5)]
Q4 = [(18, 1), (22, 5)]
QUADRANTS = [Q0, Q1, Q2, Q3, Q4]


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def pick_points(quads):
    """
        Returns a list of lists of tuples.

        quads is a list of quadrants, like [1, 2, 4, 3, 1], from which we
        will choose our random points (in the given order).

        NUM_CURVES is the number of lists;
        NUM_POINTS_IN_CURVE is the number of tuples in each list.

        Like this:

        [ [(2, 4), (7, 2), (3, 4), (1, 9)] ,
          [(3, 3), (8, 2), (5, 6), (4, 4)] ,
          [(9, 2), (7, 8), (5, 3), (3, 7)]
        ]
    """

    # results = []
    #
    # for _ in range(NUM_CURVES):

    lst = []

    for quad in quads:
        x, y = get_point_from_quadrant(QUADRANTS[quad])
        lst.append((x, y))

    return(lst)

    # results.append(lst)

    # return results

    # # This works for random points:
    # result = [np.random.rand(NUM_POINTS_IN_CURVE, 2) * MAX_COORD]
    # print(result)
    #
    # for row in range(NUM_CURVES - 1):
    #     lst = [result[row - 1][-1]]
    #     extra = np.random.rand(NUM_POINTS_IN_CURVE - 1, 2) * MAX_COORD
    #     r = np.concatenate((lst, extra), axis = 0)
    #     result.append(r)
    #
    # return result


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

    quad_list = []

    for _ in range(num_lines):

        # Choose a random number of quadrants:
        q = np.random.randint(1, 5, np.random.randint(4, 8))
        quad_list.append(q)

    return quad_list


def create_character(fname, quad_list):
    """ Creates and saves a character with filename fname. """

    # Set the filename and path:
    fname = './characters/' + fname + '.png'

    # Unpack the figure and axis, and set figure to 1 inch by 1 inch:
    fig, ax = plt.subplots(figsize = (1, 1))

    # Turn the axes off so that we only see and save the image:
    ax.set_axis_off()

    points_list = []

    for q in quad_list:

        # Grab NUM_POINTS_IN_CURVE points for each of the NUM_CURVES curves:
        points_list.append(pick_points(q))

    # Graph all of the curves:
    for lst in points_list:

        xvals, yvals = bezier_curve(lst, nTimes=1000)

        # solid_capstyle is used for capping line endpoints:
        plt.plot(xvals, yvals, 'k', linewidth = 2, solid_capstyle = 'round')

    # Some characteristics of the graph:
    plt.xlim(0, 28)
    plt.ylim(0, 28)

    # Save the figure:
    plt.savefig(fname, dpi = 28)
    plt.close()

    # Convert to reversed grayscale, so the writing is white, background black:
    # with Image.open(fname).convert("LA") as image:
    #     arr = np.asarray(image)
    #     plt.imsave(fname, arr, cmap = 'gray_r', dpi = 28)

    Image.open(fname).convert('L').save(fname)


def main():
    """ The main function. """

    asking = True

    while asking:

        # Pick a random number of lines to use, and generate quadrants for each:
        quad_list = define_lines(np.random.randint(2, 5))

        # Make one character to see if it's a keeper:
        create_character('test', quad_list)

        filename = input()

        if filename != '':
            asking = False

    for pic in range(20):

        # Create a new character:
        create_character(filename + str(pic), quad_list)


if __name__ == "__main__":
    main()
