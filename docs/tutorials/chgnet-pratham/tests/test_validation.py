import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from validation import compare, smoke

class ValidationChecks(unittest.TestCase):
    def test_wrong_shapes_and_nan_fail(self):
        self.assertFalse(compare([1,2],[1])['passed'])
        self.assertFalse(compare([np.nan],[1])['passed'])
    def test_material_error_fails(self):
        self.assertFalse(compare(-7.0,-7.36769,rtol=1e-4)['passed'])
    def test_tolerance_uses_max_not_sum(self):
        self.assertFalse(compare(1.00015,1.0,rtol=1e-4,atol=1e-4)['passed'])
        self.assertTrue(compare(1.00009,1.0,rtol=1e-4,atol=1e-4)['passed'])
    def test_smoke_rejects_wrong_force_shape(self):
        pred={'e':0.,'f':np.zeros((7,3)),'s':np.zeros((3,3)),'m':np.zeros(8)}
        self.assertFalse(smoke(pred)['f'])

if __name__=='__main__':unittest.main()
