'''
coordinate conversions
'''
import numpy as np

def cart_to_polar(x, y):
    '''
    returns the rho, phi coordinates of the x, y input points
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x, y)
    return rho, phi

def polar_to_cart(rho, phi):
    '''
    returns the x, y coordinates of the rho, phi input points
    '''
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
