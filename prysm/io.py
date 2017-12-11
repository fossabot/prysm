''' File readers (and someday, writers) for various commercial instruments
'''
import numpy as np


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
            if '>>>>>Begin Spectral Data<<<<<' in line:
                idx = i

        data_lines = txtlines[idx + 1:]
        for line in data_lines:
            wvl, v = line.split()
            wavelengths.append(float(wvl))
            values.append(float(v))

        return {
            'wvl': np.asarray(wavelengths),
            'values': np.asarray(values),
        }
