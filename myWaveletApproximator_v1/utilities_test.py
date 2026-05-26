import unittest
import numpy as np
from waveletApproximator_v2.utilities import _compute_filter_coefficients, wavelet_coefficient_generator

class TestNevillesTheorem(unittest.TestCase):
    def test_coefficients(self):
        # test to make sure that p=4 generates the correct coefficient matrix
       expected = [[5/16, 15/16, -5/16, 1/16], [-1/16,9/16,9/16,-1/16], [1/16, -5/16, 15/16, 5/16 ]]
       actual = _compute_filter_coefficients(4)

       np.testing.assert_allclose(actual,expected, atol=1e-7)

# add more cases here

class TestWaveletCoefficientFunction(unittest.TestCase):
    def test_coefficients(self):

        """ Test to make sure that for f(x) = sin(x) on domain 0, 2*pi for eps = 0.01 and p = 4, we obtain the correct wavelet coefficients on j=1 and j=2 levels """

        # expected d-coefficients for j=1 resolution level
        d1expected = np.array([0.01192335, -0.00782614, -0.00782614, -0.00324169,  0.00324169,  0.00782614, 0.00782614, -0.01192335])

        # expected d-coefficients for j=2 resolution level
        d2expected = np.array([ 0.000447,   -0.00030571, -0.00045753, -0.00053969, -0.00053969, -0.00045753,
                                    -0.00030571, -0.00010735,  0.00010735,  0.00030571,  0.00045753,  0.00053969,
                                    0.00053969, 0.00045753,  0.00030571, -0.000447  ])
        
        coarseX1 = np.linspace(0, 2*np.pi, 9)
        refinedX1 = np.linspace(0, 2*np.pi, 17)
        coarseF1 = np.sin(coarseX1); refinedF1=np.sin(refinedX1)

        coarseX2 = refinedX1
        refinedX2 = np.linspace(0, 2*np.pi, 33)
        coarseF2 = refinedF1; refinedF2 = np.sin(refinedX2)

        coef = _compute_filter_coefficients(4)

        d1, plottingF1 = wavelet_coefficient_generator(4, 1, 1, 0.00001, coef, coarseF1, refinedF1)
        d2, plottingF2 = wavelet_coefficient_generator(4, 2, 1, 0.00001, coef, coarseF2, refinedF2)

        np.testing.assert_allclose(d1expected, d1, atol=1e-7)
        np.testing.assert_allclose(d2expected, d2, atol=1e-7)