""" BWT FUNCTION """
""" 
    Builds Backward Wavelet Transform matrix 'B' for a given basis order 'p' and a given resolution 'j' 
    Applicable to the Deslauriers-Dubuc biorthogonal wavelet family
    'B' generated should be a square matrix with length 2^(j+1)*p + 1 
"""
import numpy as np

def BWT(j,p):
    hSquiggle = np.zeros((2**j*p+1,2**(j+1)*p+1))
    gSquiggle = np.zeros((2**j*p,2**(j+1)*p+1))
    

    return np.append(hSquiggle,gSquiggle,axis=0)