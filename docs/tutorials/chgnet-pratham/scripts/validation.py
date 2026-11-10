"""Checks use max(abs tolerance, relative tolerance), matching pytest.approx."""
import numpy as np

SHAPES = {'e': (), 'f': (8, 3), 's': (3, 3), 'm': (8,)}
RTOL = {'e': 1e-4, 'f': 1e-3, 's': 5e-3, 'm': 1e-3}
UNITS = {'e':'eV/atom','f':'eV/angstrom','s':'GPa','m':'mu_B'}

def smoke(prediction):
    return {key: bool(np.asarray(prediction[key]).shape == shape and
                      np.isfinite(prediction[key]).all())
            for key, shape in SHAPES.items()}

def compare(actual, expected, rtol=0, atol=1e-4):
    a,b = np.asarray(actual),np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return {'passed': False, 'reason': 'Shape mismatch or non-finite value'}
    error=np.abs(a-b)
    return {'passed': bool(np.all(error <= np.maximum(atol,rtol*np.abs(b)))),
            'max_abs_error': float(error.max()), 'rtol':rtol,'atol':atol}
