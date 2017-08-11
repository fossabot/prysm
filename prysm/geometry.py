''' Contains functions used to generate various geometrical constructs
'''
import numpy as np

def rotated_ellipse(width_major=1, width_minor=1, major_axis_angle=0, samples=128):
    '''Generates a binary mask for an ellipse, centered at the origin.  The
        major axis will notionally extend to the limits of the array, but this
        will not be the case for rotated cases.

    Args:
        width_major (`float`): width of the ellipse in its major axis.

        width_minor (`float`): width of the ellipse in its minor axis.

        major_axis_angle (`float`): angle of the major axis w.r.t. the x axis.
            specified in degrees.

        samples (`int`): number of samples

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
    major_axis_term = np.power((xv*np.cos(A)+yv*np.sin(A)),2)/a**2
    minor_axis_term = np.power((xv*np.sin(A)-yv*np.cos(A)),2)/b**2
    arr[major_axis_term + minor_axis_term > 1] = 0
    return arr
