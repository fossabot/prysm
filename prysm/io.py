''' File readers (and someday, writers) for various commercial instruments
'''
import numpy as np

from prysm.conf import config


def read_oceanoptics(file_path):
    ''' Reads spectral transmission data from an ocean optics spectrometer
        into a new Spectrum object.

    Args:
        file_path (`string`): path to a file.

    Returns:
        `Spectrum`: a new Spectrum object.

    '''
    with open(file_path, 'r') as fid:
        # txt = fid.read()
        txtlines = fid.readlines()

        wavelengths, values = [], []
        idx, ready_length, ready_spectral = None, False, False
        for i, line in enumerate(txtlines):
            if 'Number of Pixels in Spectrum' in line:
                length, ready_length = int(line.split()[-1]), True
            elif '>>>>>Begin Spectral Data<<<<<' in line:
                idx, ready_spectral = i, True

        if not ready_length or not ready_spectral:
            raise IOError('''File lacks line stating "Number of Pixels in Spectrum" or
                             ">>>>>Begin Spectral Data<<<<<" and appears to be corrupt.''')
        data_lines = txtlines[idx + 1:]
        wavelengths = np.empty(length, dtype=config.precision)
        values = np.empty(length, dtype=config.precision)
        for idx, line in enumerate(data_lines):
            wvl, val = line.split()
            wavelengths[idx] = wvl
            values[idx] = val

        return {
            'wvl': wavelengths,
            'values': values,
        }
