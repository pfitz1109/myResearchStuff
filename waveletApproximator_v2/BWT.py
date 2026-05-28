"""" BACKWARD WAVELET TRANSFORM FUNCTION """

""" 
    This function generates the 'B' matrix that converts discrete wavelet points
    to physical values. It makes use of the filter coefficient function 
    defined in the "utilities" program to construct the entries within 
    the matrix, as well as the hMatrix and gMatrix constructors in the 
    "transformUtilities" program. 
    Note that the matrix is not constructed sparsely for simplicity - this is NOT
    meant to be a computationally-efficient program, but instead one that achieves
    the correct results. 
"""

import numpy as np
from utilities import _compute_filter_coefficients
from transformUtilities import hMatrixConstructor, gMatrixConstructor

def BWT(p,J,d):
    # number of boundary conditions 
    m = int((p-2)/2)

    # generate filter coefficients
    filterCoefficients = _compute_filter_coefficients(p)

    # initialize an identity matrix for B
    B = np.eye(2**(J)*p+1)
    # will need to construct each resolution level's F matrix and assemble
    # it via repeated matrix multiplication
    for j in range(1,J):
        # create a blank identity matrix that we will fill in - bottom-right
        # corner will always be identity. just have to fill in the top-left 
        # 2^(j)*p+1 entries
        levelB = np.eye(2**(J)*p+1,2**(J)*p+1)

        # construct the h and g matrices for this resolution level
        hMatrix = hMatrixConstructor(p,j,filterCoefficients)
        gMatrix = gMatrixConstructor(p,j)

        # concatenate horizontally (hMatrix on left, gMatrix on right)
        comboMatrix = np.hstack((hMatrix,gMatrix))

        # set the upper-left quadrant of levelB to be this new matrix
        combo_rows, combo_cols = comboMatrix.shape
        levelB[:combo_rows, :combo_cols] = comboMatrix

        # update B - note that it is left-multiplied, opposite order of F
        B = levelB @ B

    # compute the f array
    f = B @ d

    return B, f