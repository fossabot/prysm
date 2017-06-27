'''
Supplimental tools for computing fourier transforms
'''

import numpy as np

def pad2d(array, factor=1):
    '''
        pads a 2D array such that the output array is 2*factor times larger in each dimmension than
        the input array
    '''
    x, y = array.shape
    pad_shape = ((x*factor, x*factor), (y*factor, y*factor))
    return np.pad(array, pad_width=pad_shape, mode='constant', constant_values=0)

def is_power_of_2(value):
    '''
    c++ inspired implementation, see SO:
    https://stackoverflow.com/questions/29480680/finding-if-a-number-is-a-power-of-2-using-recursion
    '''
    return bool(value and not value&(value-1))
