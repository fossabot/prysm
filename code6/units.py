'''Unit manipulation
'''

def waves_to_microns(wavelength):
    '''Returns a conversion factor to yield microns from OPD expressed in waves

    Args:
        wavelength (float): wavelength of light, expressed in microns

    Returns:
        conversion factor from waves to microns
    '''
    return 1/wavelength

def waves_to_nanometers(wavelength):
    '''Returns a conversion factor to yield nanometers from OPD expressed in waves

    Args:
        wavelength (float): wavelength of light, expressed in microns

    Returns:
        conversion factor from waves to nanometers
    '''
    return 1/(wavelength*1e3)

def microns_to_waves(wavelength):
    '''Returns a conversion factor to yield waves from OPD expressed in microns

    Args:
        wavelength (float): wavelength of light, expressed in microns

    Returns:
        conversion factor from microns to waves
    '''
    return wavelength

def nanometers_to_waves(wavelength):
    '''Returns a conversion factor to yield waves from OPD expressed in nanometers

    Args:
        wavelength (float): wavelength of light, expressed in microns

    Returns:
        conversion factor from nanometers to waves
    '''
    return wavelength*1e3
