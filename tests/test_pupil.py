''' Unit tests for pupil objects
'''
from prysm import Pupil

class TestPupil(object):
    def test_create(self):
        p = Pupil()
        assert hasattr(p, 'wavelength')
        assert hasattr(p, 'epd')
        assert hasattr(p, 'sample_spacing')
        assert hasattr(p, 'samples')
        assert hasattr(p, 'opd_unit')
        assert hasattr(p, '_opd_unit')
        assert hasattr(p, '_opd_str')
        assert hasattr(p, 'phase')
        assert hasattr(p, 'fcn')
        assert hasattr(p, 'unit')
        assert hasattr(p, 'rho')
        assert hasattr(p, 'phi')
        assert hasattr(p, 'center')
