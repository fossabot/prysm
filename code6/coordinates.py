'''
coordinate conversions
'''
import numpy as np
from scipy import interpolate

def cart_to_polar(x, y):
    '''returns the rho, phi coordinates of the x, y input points

    Args:
        x (float): x coordinate
        y (float): y coordinate

    Returns:
        float.  radial coordinate
        float.  azimuthal coordinate
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x, y)
    return rho, phi

def polar_to_cart(rho, phi):
    '''returns the x, y coordinates of the rho, phi input points

    Args:
        rho (float): radial coordinate
        phi (float): azimuthal cordinate

    Returns:
        float.  x coordinate
        float.  y coordinate
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

def uniform_cart_to_polar(x, y, data):
    '''interpolates data uniformly sampled in cartesian coordinates to polar coordinates.

    Args:
        x (numpy.array): *sorted* 1D array of x values
        y (numpy.array):
        data (numpy.array): data sampled over the x, y coordinates

    Returns:
        numpy.array.  Rho samples for interpolated values
        numpy.array.  Phi samples for interpolated values
        numpy.array.  Data uniformly sampled in (rho,phi).
    '''
    # create a set of polar coordinates to interpolate onto
    xmin, xmax = x[0], x[-1]
    num_pts = len(x)
    center = int(np.floor(num_pts/2))
    rho = np.linspace(xmin, xmax, num_pts)
    phi = np.linspace(0, 2*np.pi, num_pts)
    rv, pv = np.meshgrid(rho, phi)

    # map points to x, y and make a grid for the original samples
    xv, yv = polar_to_cart(rv, pv)

    # 3 - interpolate the function onto the new points
    f = interpolate.RegularGridInterpolator((x, y), data)
    return rv, pv, f((xv, yv), method='linear')