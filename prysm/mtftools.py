''' Tools for formatting MTF data.
'''

import numpy as np
import pandas as pd

def mtf_tan_sag_to_dataframe(tan, sag, freqs, field=0, focus=0):
    ''' Creates a Pandas dataframe from tangential and sagittal MTF data.

    Args:
        tan (`numpy.ndarray`): vector of tangential MTF data.

        sag (`numpy.ndarray`): vector of sagittal MTF data.

        freqs (`iterable`): vector of spatial frequencies for the data.

        field (`float`): relative field associated with the data.

        focus (`float`): focus offset (um) associated with the data.

    Returns:
        pandas dataframe.

    '''
    rows = []
    for f, s, t in zip(freqs, tan, sag):
        base_dict = {
            'Field': field,
            'Focus': focus,
            'Freq': f,
        }
        rows.append({**base_dict, **{
            'Azimuth': 'Tan',
            'MTF': t,
            }})
        rows.append({**base_dict, **{
            'Azimuth': 'Sag',
            'MTF': s,
            }})
    return pd.DataFrame(data=rows)
