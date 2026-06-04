""" MAIN FILE FOR RUNNING WAVELET APPROXIMATOR """

""" 
    This function generates the dense wavelet approximation for a 1D funciton/
    signal. It makes use of the FWT and BWT functions, which generate the 'F'
    and 'B' matrices, to convert back-and-forth between physical space and 
    wavelet space. This will create a dense grid of collacation points up to
    a maximum resolution level 'J.' The thresholding value 'eps' determines 
    to what accuracy we wish to approximate our signal - for a sufficiently
    high resolution level, we can guarantee that our approximated signal will
    be at least eps-order accurate. 

    Inputs: 
    1. p - basis order. Determines number of interpolation points and continuity
    of approximation.
    2. eps - thresholding value. Any wavelet coefficients with magnitude less than
    this value are set equal to zero.
    3. J - maximum resolution level. Matrix operators are created based on this
    value. 
    4. func - input signal to be transformed/approximated.
    5. left_bound - left-boundary coordinate of domain.
    6. right_bound - right-boundary coordinate of domain.
    7. status_update - Boolean that, when turned on, prints out status updates 
    of each function. By default, set equal to 'False' so output is not crowded.

    Outputs:
    1. F - forward wavelet transform (FWT) matrix. Constructed up to resolution
    level 'J'.
    2. B - backward wavelet transform (BWT) matrix. Constructed up to resolution
    level 'J'. 
    3. s0 - s0 coefficients. Corresponds to evaluation of function on coarsest
    grid (i.e., grid with 2*p+1 collocation points)
    4. coefficientThresholdArray - thresholded coefficients. Removes s0 from 'd'
    array generated in FWT, so it is just the wavelet coefficients at the 
    colllocation points.
    5. completeCoefficientArray - s0 + coefficientThresholdArray. 
    6. fApproximate - approximated function vector. Computed via BWT of the 
    'completeCoefficientArray.'
"""

import numpy as np
import matplotlib.pyplot as plt

from FWT import FWT
from BWT import BWT
from utilities import _validate_eps, _validate_p
from transformUtilities import thresholdCoefficients

def _dense_wavelet_approximation_function(p,eps,J,func,left_bound,right_bound,status_update=False):
    print('\n')
    print(fr'### APPROXIMATING USING p={p}, J={J}, eps={eps} ###')
    """ PRELIMINARY TESTS """
    # checking to make sure that inputs are valid
    _validate_eps(eps); _validate_p(p)

    """ CREATE THE FWT AND BWT MATRICES """
    """
        The way that i have written this, it will automatically create the matrix F
        but will also generate the vector of coefficients 'd'.
    """
    F, d = FWT(p, J, func, left_bound, right_bound,status_update)

    """ THRESHOLD THE WAVELET COEFFICIENTS """
    """
        To generate an approximation, you have to threshold the 'd' coefficients 
        to the thresholding value 'eps.' The below function returns the s0 coeffs
        (the exact function values on the coarsest grid, which usees 2*p+1 points),
        the thresholded coefficients (removes s0 coefficients and thresholds), and 
        the "complete" d-vector (s0 coefficients, followed by thresholded values). 
        The completeCoefficientArray is what we will use to perform the BWT. 
    """

    s0, coefficientThresholdArray, completeCoefficientArray = thresholdCoefficients(p, d, eps, status_update)

    """ BACKWARD WAVELET TRANSFORMATION """
    """
        We now take the 'completeCoefficientArray' and pass it through the BWT 
        operator to generate our approximation up to an order of accuracy of 'eps.'
        Returns the 'B' matrix as well as the approximated function on the finest
        grid resolution.
    """
    B, fApproximate = BWT(p, J, completeCoefficientArray, status_update)
    return F, B, s0, coefficientThresholdArray, completeCoefficientArray, fApproximate