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
        idx = None
        for i, line in enumerate(txtlines):
            if 'Number of Pixels in Spectrum' in line:
                length = int(line.split()[-1])
            elif '>>>>>Begin Spectral Data<<<<<' in line:
                idx = i

        data_lines = txtlines[idx + 1:]
        wavelengths = np.empty(length, dtype=config.precision)
        values = np.empty(length, dtype=config.precision)
        for idx, line in enumerate(data_lines):
            wvl, v = line.split()
            wavelengths[idx] = wvl
            values[idx] = v

        return {
            'wvl': wavelengths,
            'values': values,
        }
