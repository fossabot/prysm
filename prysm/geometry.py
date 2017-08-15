''' Contains functions used to generate various geometrical constructs
'''
import numpy as np
from numpy import power as npow

def gaussian(sigma=0.5, samples=128):
    ''' Generates a gaussian mask with a given sigma

    Args:
        sigma (`float`): width parameter of the gaussian, expressed in radii of
            the output array.

        samples (`int`): number of samples in square array.

    Returns:
        `numpy.ndarray`: mask with gaussian shape.

    '''
    s = sigma

    x = np.arange(0, samples, 1, float)
    y = x[:, np.newaxis]

    # // is floor division in python
    x0 = y0 = samples // 2
    return np.exp(-4*np.log(2) * (npow(x-x0,2) + npow(y-y0,2)) / (s*samples)**2)

def rotated_ellipse(width_major, width_minor, major_axis_angle=0, samples=128):
    ''' Generates a binary mask for an ellipse, centered at the origin.  The
        major axis will notionally extend to the limits of the array, but this
        will not be the case for rotated cases.

    Args:
        width_major (`float`): width of the ellipse in its major axis.

        width_minor (`float`): width of the ellipse in its minor axis.

        major_axis_angle (`float`): angle of the major axis w.r.t. the x axis.
            specified in degrees.

        samples (`int`): number of samples.

    Returns:
        numpy.ndarray: an ndarray of shape (samples,samples) of value 0 outside
            the ellipse, and value 1 inside the ellipse.

    Notes:
        The formula applied is:
             ((x-h)cos(A)+(y-k)sin(A))^2      ((x-h)sin(A)+(y-k)cos(A))^2
            ______________________________ + ______________________________ 1
                         a^2                               b^2
        where x and y are the x and y dimensions, A is the rotation angle of the
        major axis, h and k are the centers of the the ellipse, and a and b are
        the major and minor axis widths.  In this implementation, h=k=0 and the
        formula simplifies to:
                (x*cos(A)+y*sin(A))^2             (x*sin(A)+y*cos(A))^2
            ______________________________ + ______________________________ 1
                         a^2                               b^2

        see SO:
        https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate

    '''
    if width_minor > width_major:
        raise ValueError('by definition, major axis must be larger than minor.')

    arr = np.ones((samples, samples))
    lim = width_major
    x, y = np.linspace(-lim, lim, samples), np.linspace(-lim, lim, samples)
    xv, yv = np.meshgrid(x, y)
    A = np.radians(-major_axis_angle)
    a, b = width_major, width_minor
    major_axis_term = np.power((xv*np.cos(A) + yv*np.sin(A)),2) / a**2
    minor_axis_term = np.power((xv*np.sin(A) - yv*np.cos(A)),2) / b**2
    arr[major_axis_term + minor_axis_term > 1] = 0
    return arr

def triangle(samples=128):
    ''' Creates a triangular mask.
    '''
    return regular_polygon_mask(3, samples)

def square(samples=128):
    ''' Creates a square mask.
    '''
    return regular_polygon_mask(4, samples)

def pentagon(samples=128):
    ''' Creates a pentagonal mask.
    '''
    return regular_polygon_mask(5, samples)

def hexagon(samples=128):
    ''' Creates a hexagonal mask.
    '''
    return regular_polygon_mask(6, samples)

def heptagon(samples=128):
    ''' Creates a heptagonal mask.
    '''
    return regular_polygon_mask(7, samples)

def octagon(samples=128):
    ''' Creates an octagonal mask.
    '''
    return regular_polygon_mask(8, samples)

def nonagon(samples=128):
    ''' Creates a nonagonal mask.
    '''
    return regular_polygon_mask(9, samples)

def decagon(samples=128):
    ''' Creates a decagonal mask.
    '''
    return regular_polygon_mask(10, samples)

def hendecagon(samples=128):
    ''' Creates a hendecagonal mask.
    '''
    return regular_polygon_mask(11, samples)

def dodecagon(samples=128):
    ''' Creates a dodecagonal mask.
    '''
    return regular_polygon_mask(12, samples)

def trisdecagon(samples=128):
    ''' Creates a trisdecagonal mask.
    '''
    return regular_polygon_mask(13, samples)

def regular_polygon_mask(num_sides, num_samples):
    ''' Generates a regular polygon mask with the given number of sides and
        samples in the mask array.

    Args:
        num_sides (`int`): number of sides to the polygon.

        num_samples (`int`): number of samples in the output polygon.

    Returns:
        `numpy.ndarray`: mask for regular polygon with radius equal to the array
            radius.

    '''
    verts = generate_vertices(num_sides, num_samples/2)
    verts[:,0] += num_samples/2 # shift y to center
    verts[:,1] += num_samples/2 # shift x to center
    return generate_mask(verts, num_samples)

def check(p1, p2, base_array):
    ''' Checks if the values in the base array fall inside of the triangle
        enclosed in the points (p1, p2, (0,0)).

    Args:
        p1 (`iterable`): iterable containing (x,y) coordinates of a point.

        p2 (`iterable`): iterable containing (x,y) coordinates of a point.

        base_array (`numpy.ndarray`): a logical array.

    Returns:
        `numpy.ndarray`: array with True value inside and False value outside bounds

    '''
    # Create 3D array of indices
    idxs = np.indices(base_array.shape)

    # ensure points are floats
    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

def generate_mask(vertices, num_samples=128):
    ''' Creates a filled convex polygon mask based on the given vertices.

    Args:
        vertices (`iterable`): ensemble of vertice (x,y) coordinates, in array units.

        num_samples (`int`): number of points in the output array along each dimension.

    Returns:
        `numpy.ndarray`: polygon mask.

    Notes:
        Stolen from:
        https://stackoverflow.com/questions/37117878/generating-a-filled-polygon-inside-a-numpy-array
    '''
    shape = (num_samples, num_samples)
    vertices = np.asarray(vertices)

    # Initialize background and mask arrays
    base_array = np.zeros(shape, dtype=float)
    fill = np.ones(base_array.shape) * True

    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)

    # Set all values inside polygon to one
    base_array[fill] = 1

    return base_array

def generate_vertices(num_sides, radius=1):
    ''' Generates a list of vertices for a convex regular polygon with the given
        number of sides and radius.

    Args:
        num_sides (`int`): number of sides to the polygon.

        radius (`float`): radius of the polygon.

    Returns:
        `numpy.ndarray`: array with first column X points, second column Y points

    '''
    angle = 2 * np.pi / num_sides
    pts = []
    for point in range(num_sides):
        x = radius * np.sin(point * angle)
        y = radius * np.cos(point * angle)
        pts.append((x,y))
    return np.asarray(pts)
