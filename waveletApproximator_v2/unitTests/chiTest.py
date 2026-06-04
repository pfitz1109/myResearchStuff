""" CHI TEST """
"""
    Tests to make sure that the _chi_constructor function proudces the correct
    eigenvector 'chi' for a given basis order 'p', derivative order 'a', and
    left and right bounds. 

    These values are compared to those produced by the Mathematica MRWT toolbox.
    All test values are compared to the 'interior' function found within that 
    toolbox, which generates the normalized eigenvector. 
"""

import unittest
import numpy as np
from derivativeUtilities import _chi_constructor

class chiTest(unittest.TestCase):
    def test_matrix(self):
        # p=4, a=1 case
        chiOneTest = _chi_constructor(4,1,0,2*np.pi)
        chiOneActual = [0,0,-1/(3*np.pi),8/(3*np.pi),0,-8/(3*np.pi),1/(3*np.pi),0,0]
        np.testing.assert_allclose(chiOneTest, chiOneActual, atol=1e-7)

        # p=6, a=2 case
        chiTwoTest = _chi_constructor(6,2,0,2*np.pi)
        chiTwoActual = [0,0,27/(140*np.pi**2),144/(35*np.pi**2),-1104/(35*np.pi**2),
                        4272/(35*np.pi**2),-2655/(14*np.pi**2), 4272/(35*np.pi**2), 
                        -1104/(35*np.pi**2),144/(35*np.pi**2),27/(140*np.pi**2),0,0]
        np.testing.assert_allclose(chiTwoTest, chiTwoActual, atol=1e-7)

        # p=8, a =3 case
        chiThreeTest = _chi_constructor(8,3,0,2*np.pi)
        denom = (np.pi**3)
        chiThreeActual = [0,0,-120/(2611*denom),6144/(2611*denom),-87616/(39165*denom),
                          -1271808/(13055*denom),24383896/(39165*denom),-37421056/(39165*denom),
                          0,37421056/(39165*denom),-24383896/(39165*denom),1271808/(13055*denom),
                          87616/(39165*denom),-6144/(2611*denom),120/(2611*denom),0,0]
        np.testing.assert_allclose(chiThreeTest, chiThreeActual, atol=1e-7)

        # p=10, a=4 case
        chiFourTest = _chi_constructor(10,4,0,2*np.pi)
        denom = 1/(np.pi**4)
        chiFourActual = denom*np.array([0,0,7015625/13673232,359200000/5982039,-1125233750/664671,
                         46193120000/5982039, 36877608625/23928156,-74879584000/664671,
                         2601185279750/5982039,-5295054752000/5982039,63505625625/57176,
                         -5295054752000/5982039,2601185279750/5982039,-74879584000/664671,
                         36877608625/23928156,46193120000/5982039,-1125233750/664671,
                         359200000/5982039,7015625/13673232,0,0])
        np.testing.assert_allclose(chiFourTest, chiFourActual, atol=1e-7)