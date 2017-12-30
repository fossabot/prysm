''' Tests for the colorimetry submodule
'''
import pytest

from prysm import colorimetry


PRECISION = 1e-1


def test_can_prepare_cmf_1931_2deg():
    ''' Trusts observer is properly formed.
    '''
    obs = colorimetry.prepare_cmf('1931_2deg')
    assert obs


def test_can_prepare_cmf_1964_10deg():
    ''' Trusts observer is properly formed.
    '''
    obs = colorimetry.prepare_cmf('1964_10deg')
    assert obs


def test_cmf_is_valid():
    ''' Tests if a cmf returns as valid data.
    '''
    obs = colorimetry.prepare_cmf()
    assert 'X' in obs
    assert 'Y' in obs
    assert 'Z' in obs
    assert 'wvl' in obs
    assert len(obs['X']) == len(obs['Y']) == len(obs['Z']) == len(obs['wvl'])


def test_can_get_roberson_cct():
    ''' Trusts data is properly formed.
    '''
    cct_data = colorimetry.prepare_robertson_cct_data()
    assert cct_data


def test_robertson_cct_is_valid():
    ''' Tests if the Roberson CCT data is returned properly.
    '''
    cct = colorimetry.prepare_robertson_cct_data()
    assert len(cct['urd']) == 31
    assert len(cct['K']) == 31
    assert len(cct['u']) == 31
    assert len(cct['v']) == 31
    assert len(cct['dvdu']) == 31


@pytest.mark.parametrize('illuminant', [
    'A', 'B', 'C', 'E',
    'D50', 'D55', 'D65',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
    'HP1', 'HP2', 'HP3', 'HP4', 'HP5'])
def test_can_get_illuminant(illuminant):
    ill_spectrum = colorimetry.prepare_source_spd(illuminant)
    assert ill_spectrum


def test_blackbody_spd_correctness():
    pass


def test_cct_duv_to_uvprime():
    cct = 2900  # values from Ohno 2013
    duv = 0.0200

    true_u = 0.247629
    true_v = 0.367808

    up, vp = colorimetry.CCT_Duv_to_uvprime(cct, duv)
    v = vp / 1.5
    u = up
    assert u == pytest.approx(true_u, rel=PRECISION, abs=PRECISION)
    assert v == pytest.approx(true_v, rel=PRECISION, abs=PRECISION)
