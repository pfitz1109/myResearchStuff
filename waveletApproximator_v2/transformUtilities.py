""" TRANSFORM UTILITIES """

""" 
    Contains functions that perform various tasks for generating wavelet
    transformation matrices 'F' and 'B'. Included in this program are 

    1. hTildeMatrixConstructor - constructs the hTilde matrix, given basis 
    'p' and resolution level 'j'
    2. gTildeMatrixConstructor - constructs the gTilde matrix, given basis
    'p', resolution level 'j', and filter coefficient matrix 'filterCoefficients'
    3. hMatrixConstructor - constructs the hMatrix, given basis 'p', 
    resolution level 'j', and filter coefficient matrix 'filterCoefficients'
    4. gMatrixConstructor - constructs the gMatrix, given basis 'p; and 
    resolution level 'j'
    5. thresholdCoefficients - thresholds the wavelet coefficients 'd' to a user-
    specified value 'eps'
"""

import numpy as np

# constructs the hTilde matrix for a given resolution level j and basis order p
def hTildeMatrixConstructor(j,p):
    # given definition 
    num_rows = 2**j*p+1; num_cols = 2**(j+1)*p+1
    hTildeMatrix = np.zeros((num_rows, num_cols))
    hTildeMatrix[:, ::2] = np.eye(num_rows)
    return hTildeMatrix

# constructs the gTilde matrix for a given resolution level j and basis order p
# and filterCoefficient matrix (generated from Neville's theorem above)
def gTildeMatrixConstructor(j,p, filterCoefficients):
    # given definition
    num_rows = 2**j*p; num_columns = 2**(j+1)*p+1

    # grab the number of rows in the filterCoefficients matrix - will tell you
    # how many boundary conditions there are
    filterCoefficientsSize = filterCoefficients.shape[0]

    # determine number of boundary coefficient rows
    m = int((filterCoefficientsSize-1)/2)

    # generat a blank matrix that we are going to fill in
    gTildeMatrix = np.zeros((num_rows,num_columns))

    # have to go row-by-row and assign values to each row
    for r in range(0,num_rows):
        # assign left-boundary rows 
        if r < m:
            gTildeMatrix[r, 0:(2*p-1):2 ] = filterCoefficients[r, :]
            # needs an entry of -1 at the point to be interpolated
            gTildeMatrix[r, 2*r+1] = -1 

        # right-boundary rows - slightly more tricky since we have to compute
        # the starting column a priori
        elif r > num_rows - (m+1) :
            col_start = num_columns - (2*p-1)
            # don't ask me how i got the indexing formula
            gTildeMatrix[r, col_start : col_start + (2*p-1) : 2 ] = filterCoefficients[2*m+1-(num_rows-r), :]
            # needs an entry of -1 at the point to be interpolated
            gTildeMatrix[r, 2*r+1] = -1 

        # interior rows
        else:
            start_column = 2 * (r-m)
            end_column = 2 * (r-m) + 2*p-1
            gTildeMatrix[r, start_column:end_column:2] = filterCoefficients[m,:]
            # needs an entry of -1 at the point to be interpolated 
            gTildeMatrix[r, 2*r+1] = -1

    return gTildeMatrix

def gMatrixConstructor(p,j) :
    # given definition
    num_rows = 2**(j+1)*p + 1; num_cols = 2**j*p

    # create blank matrix that we will fill in later
    gMatrix = np.zeros((num_rows, num_cols))

    # every other row is blank
    gMatrix[1: 1 + 2*num_cols:2, :] = -np.eye(num_cols)
    
    return gMatrix

# follows a very similar to structure to how the gTilde matrix is constructed,
# but done so in the "vertical" direction instead
def hMatrixConstructor(p,j,filterCoefficients):
    # given definition
    num_rows = 2**(j+1)*p+1; num_cols = 2**j*p + 1

    # grab the number of rows in the filterCoefficients matrix - will tell you
    # how many boundary conditions there are
    filterCoefficientsSize = filterCoefficients.shape[0]

    # determine number of boundary coefficient rows
    m = int((filterCoefficientsSize-1)/2)

    # generate a blank matrix that we are going to fill in
    hMatrix = np.zeros((num_rows,num_cols))

    # assign unity values where necessary
    for c in range(num_cols):
        hMatrix[2*c, c] = 1

    # diagram in "How to Wavelet" is misleading - see my own resource for why
    # we construct the matrix this way
    num_odd_rows = 2**j*p
    for r in range(num_odd_rows):
        
        # Left-boundary conditions
        if r < m:
            hMatrix[2*r+1, 0:p] = filterCoefficients[r, :]
            
        # Right-boundary conditions
        elif r > num_odd_rows - (m+1):
            hMatrix[2*r+1, num_cols-p : num_cols] = filterCoefficients[2*m+1 - (num_odd_rows - r), :]
            
        # Interior conditions
        else: 
            start_c = r - m
            hMatrix[2*r+1, start_c : start_c + p] = filterCoefficients[m, :]

    return hMatrix


# apply thresholding parameter to d-coefficients
def thresholdCoefficients(p, d, eps, status_updates=False):
    # status update
    if status_updates:
        print('Thresholding coefficients...')

    # create a copy of d
    dThreshold = np.copy(d)

    # need to remove the s0 coefficients, they should not be thresholded
    s0 = dThreshold[:2*p+1]
    coefficientThreshold = dThreshold[2*p+1:]

    # apply thresholding value 
    coefficientThreshold[np.abs(coefficientThreshold) < eps] = 0

    # add the s0 coefficients back
    dComplete = np.concatenate((s0,coefficientThreshold))

    if status_updates:
        print('Thresholding Complete.')
    # return the s0 coefficients, the thresholded d-coefficients, and the 
    # s0+thresholded coefficients (to be used in the backward transform)
    return s0, coefficientThreshold, dComplete