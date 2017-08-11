''' A collection of thin lens equations for system modeling
'''
import numpy as np
from numpy import sin, arctan, absolute, sqrt, pi
from prysm.util import guarantee_array
from prysm.fringezernike import _normalizations

def object_to_image_dist(efl, object_distance):
    '''computes the image distance from the object distance

    Args:
        efl (float): focal length of the lens
        object_distance (float or numpy.ndarray): distance from the object to the front principal
            plane of the lens, negative for an object to the left of the lens.

    Returns:
        image distance.  Distance from rear principal plane (assumed to be in contact with front
            principal plane) to image

    '''
    object_distance = guarantee_array(object_distance)
    ret = 1 / efl - 1 / object_distance
    return 1 / ret

def image_dist_epd_to_na(image_distance, epd):
    '''Computes the NA from an image distance and entrance pupil diameter

    Args:
        image_distance (float): distance from the image to the entrance pupil
        epd (float): diameter of the entrance pupil

    Returns:
        numerical aperture.  The NA of the system.

    '''
    image_distance = guarantee_array(image_distance)

    rho = epd / 2
    marginal_ray_angle = absolute(arctan(rho/image_distance))
    return marginal_ray_angle

def image_dist_epd_to_fno(image_distance, epd):
    '''Computes the f/# from an image distance and entrance pupil diameter

    Args:
        image_distance (float): distance from the image to the entrance pupil
        epd (float): diameter of the entrance pupil

    Returns:
        fno.  The working f/# of the system.

    '''
    na = image_dist_epd_to_na(image_distance, epd)
    return na_to_fno(na)

def na_to_fno(na):
    '''Converts an NA to an f/#

    Args:
        na (float): numerical aperture

    Returns:
        fno.  The f/# of the system.

    '''
    return 1 / (2 * sin(na))

def mag_from_object_dist(efl, object_dist):
    '''Computes the linear magnification from the object distance and focal length

    Args:
        efl (float): focal length of the lens
        object_dist (float): object distance

    Returns:
        linear magnification.  Also known as the lateral magnification.
    
    '''
    object_dist = guarantee_array(object_dist)
    return efl / (efl - object_dist)

def logitudinal_mag_from_linear(lateral_mag):
    '''Computes the longitudinal (along optical axis) magnification from the lateral mag

    Args:
        lateral_mag (float): linear magnification, from thin lens formulas

    Returns:
        longitudinal magnification.

    '''
    return lateral_mag**2

def mag_to_fno(mag, infinite_fno, pupil_mag=1):
    '''Computes the working f/# from the magnification and infinite f/#

    Args:
        mag (float or numpy.ndarray): linear or lateral magnification
        infinite_fno (float): f/# as defined by EFL/EPD
        pupil_mag (float): pupil magnification

    Returns:
        working fno.

    '''
    mag = guarantee_array(mag)
    return (1 + np.absolute(mag) / pupil_mag) * infinite_fno

def defocus_to_image_displacement(defocus, fno, wavelength, zernike=False, rms_norm=False):
    '''Computes image displacment from wavefront defocuse xpressed in waves 0-P to

    Args:
        defocus (float or numpy.ndarray): wavefront defocus

        fno (float): f/# of the lens or system

        wavelength (float): wavelength of light, expressed in micron

        zernike (bool): zernike model of defocus (otherwise model is Seidel)

        rms_norm (bool): if zernike model, term is rms normalized

    Returns:
        image displacement.  Motion of image in um caused by defocus OPD

    '''
    defocus = guarantee_array(defocus)

    # if the defocus is a zernike, make it match Seidel notation for equation validity
    if zernike is True:
        if rms_norm is True:
            defocus /= (eval(_normalizations[4]) * 2)
        else:
            defocus /= 2
    return 8 * fno**2 * wavelength * defocus

def image_displacement_to_defocus(image_displacement, fno, wavelength, zernike=False, rms_norm=False):
    '''Computes the wavefront defocus from image shift, expressed in the same units as the shift

    Args:
        image_displacement (`float` or ~`numpy.ndarray`): displacement of the image

        fno (`float`): f/# of the lens or system

        wavelength (`float`): wavelength of light, expressed in microns

        zernike (`bool`): return in Zernike notation

        rms_norm (`bool`): subset of zernike -- return rms normalized zernike

    Returns:
        `float`. wavefront defocus

    '''
    image_displacement = guarantee_array(image_displacement)
    defocus = image_displacement / (8 * np.power(fno,2) * wavelength)
    if zernike is True:
        if rms_norm is True:
            return defocus / 2 / eval(_normalizations[4])
        else:
            return defocus / 2
    else:
        return defocus
