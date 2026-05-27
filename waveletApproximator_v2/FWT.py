""" FORWARD WAVELET TRANSFORM (FWT) FUNCTION"""

"""
    This function generates the 'F' matrix that converts discrete physical points
    to wavelet coefficients. It makes use of the filter coefficient function 
    defined in the "utilities" program to construct the entries within the matrix.
    Note that the matrix is not constructed sparsely for simplicity - this is NOT
    meant to be a computationally-efficient program, but instead one that achieves
    the correct results. 
"""

""" 
    Some helpful definitions:

    h_k : [...0 0 0 h1 0 h2 0 ... 0 h(p/2-1) 1 h(p/2+1) 0 ... 0 h(p-1) 0 hp 0 0 0 ...] (these coefficients get multiplied by previous resolultion level)
    hS_k = [... 0 0 0 1 0 0 0 ...] (only applies at current index location 'k') 
    g_{k+1} = (-1)^{k+1} * hS_{-k}
    gS_{k+1} = (-1)^{k+1} * h_{-k}

"""

"""
    INPUTS:
    p - basis order (even, positive integer leq 10)
    J - maximum resolution level (even, positive integer)
    func - input signal/function (what is to to be transformed)
    left_bound - left-boundary of x-domain
    right_bound - right-boundary of x-domain

    OUTPUTS:
    F - forward wavelet transform matrix, dimensions: (2^(J)*p+1) X (2^(J)*p+1)
    d - numpy array containing all wavelet coefficients up to resolution level J
"""

import numpy as np
from utilities import _compute_filter_coefficients, hTildeMatrixConstructor, gTildeMatrixConstructor

def FWT(p,J,func,left_bound,right_bound) :
    # number of boundary conditions
    m = int((p-2)/2)

    # create an empty container to store the wavelet coefficients
    d = []

    # discretize signal on finest resolution
    finestX = np.linspace(left_bound,right_bound,2**(J)*p+1)
    evalF = func(finestX)

    # generate filter coefficients 
    filterCoefficients = _compute_filter_coefficients(p)

    # generate hTilde vectors 

    # initialize an identity matrix as F
    F = np.eye(2**(J)*p+1)
    # think that you're going to need a for-loop to construct the F matrix
    for j in range(1,J):
        # create a blank identity to fill in - bottom-right corner will always
        # be an identity matrix. just have to fill in the top-left 2^(j)*p+1 entries
        levelF = np.eye(2**(J)*p+1,2**(J)*p+1)

        # construct the hTilde and gTilde matrices for this resolution level
        hTilde = hTildeMatrixConstructor(j,p)
        gTilde = gTildeMatrixConstructor(j,p,filterCoefficients)

        # concatenate (put the hTilde on top, gTilde on bottom - vertically stack)
        comboMatrix = np.vstack((hTilde,gTilde))

        # set the upper-left quadrant of levelF to be this new matrix
        combo_rows, combo_cols = comboMatrix.shape
        levelF[:combo_rows, :combo_cols] = comboMatrix

        # update F 
        F = F @ levelF
    
    # compute the d array 
    d = F @ evalF
    
    return F, d