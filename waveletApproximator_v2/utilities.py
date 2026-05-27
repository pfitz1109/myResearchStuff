""" Utility functions for sparseWaveletApproximator.py and denseWaveletApproximator.py """
import numpy as np

# validates p, checks that it is a positive even integer less than or equal to 10
def _validate_p(p: int) -> None:
    if p <=0 :
        raise ValueError(f'p must be a positive integer, got p={p}')
    if p % 2 != 0:
        raise ValueError(f'p must be an even integer, got p={p}')
    if p > 10:
        raise ValueError(f'Code can only handle interpolation orders up to p=10, please try a smaller interpolation order. Got p={p}')

# validates eps, checks that it is a positive float 
def _validate_eps(eps: float) -> None: 
    if eps <= 0:
        raise ValueError(f'Thresholding value "eps" must be greater than zero, got {eps}')

# neville's theorem for computing the filter coefficients h
def _compute_filter_coefficients(p: int) -> np.ndarray:
    coef = np.ones((p-1,p))

    for i in range (0,p-1):
        for j in range(0,p):
            for k in range(0,p):
                if (k==j):
                    continue
                coef[i,j] = coef[i,j]*(i+0.5-k)/(j-k)
    return coef

# constructs the hTilde matrix for a given resolution level j and basis order p
def hTildeMatrixConstructor(j,p):
    hTildeMatrix = np.zeros((2**j*p+1, 2**(j+1)*p+1))
    hTildeMatrix[:, ::2] = np.eye(2**j*p+1)
    return hTildeMatrix

# constructs the gTilde matrix for a given resolution level j and basis order p
# and filterCoefficient matrix (generated from Neville's theorem above)
def gTildeMatrixConstructor(j,p, filterCoefficients):
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
            row_start = num_columns - (2*p-1)
            # don't ask me how i got the indexing formula
            gTildeMatrix[r, row_start : row_start + (2*p-1) : 2 ] = filterCoefficients[2*m+1-(num_rows-r), :]
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

# apply thresholding parameter to d-coefficients
def thresholdCoefficients(d, eps):
    dThreshold = np.copy(d)
    return