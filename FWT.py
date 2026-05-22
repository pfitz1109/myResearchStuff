""" FWT FUNCTION """
""" 
    Builds Forward Wavelet Transform matrix 'F' for a given basis order 'p' and a given resolution 'j' 
    Applicable to the Deslauriers-Dubuc biorthogonal wavelet family
    'F' generated should be a square matrix with length 2^(j+1)*p + 1 
"""

import numpy as np
from utilities import _compute_filter_coefficients

def FWT(p,j):
    hSquiggle = np.zeros((2**j*p+1,2**(j+1)*p+1))
    gSquiggle = np.zeros((2**j*p,2**(j+1)*p+1))
    
    return np.append(hSquiggle,gSquiggle,axis=0)