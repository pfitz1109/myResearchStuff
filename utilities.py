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

def wavelet_coefficient_generator (p, j, m, eps, coef, coarseF, refinedF):

    # number of midpoint nodes generated for the current resolution level
    N = 2**(j)*p

    # initialize d vector, will store thresholding coefficients in here
    d = np.zeros(N)    

    # initialize this array to store the function values we will plot later
    plottingF = np.zeros(N)

    # going through every collocation point in the refined grid
    for k in range(0,N):
        """ General outline of the computation: 
            1. Compute the fSquiggle vector (previous grid's function values at neighboring nodes)
            2. Compute d using d = dot(filterCoefficients,coarseF[neighboringPoints]) - refinedF[currentNode]
            3. Perform thresholding conditional - if the magnitude of the coefficient is less than epsilon, set it equal to zero
            4. Compute the approximation of the function by using f ~= dot(filterCoefficients,fcoarseF[neighboringPoints]) - d
                Note that if d=0, we generate an approximation - it is not exact 
            Perform these four steps at each collocation point - boundary points (which have an index of 0 -> k and N-k -> N) get
            special treatment 
        """
        # left boundary condition(s)
        if k < m:
            fSquiggle = coarseF[0:p]
            d[k] = np.dot(fSquiggle,coef[k,:]) - refinedF[2*k+1]
            if abs(d[k]) <= eps:
                d[k] = 0
            plottingF[k] = np.dot(fSquiggle,coef[k,:]) - d[k]

        # right boundary condition(s)
        elif k > N-(m+1):
            fSquiggle = coarseF[-p:]
            d[k] = np.dot(fSquiggle,coef[2*m+1-(N-k),:]) - refinedF[2*k+1]
            if abs(d[k]) <= eps:
                d[k] =0
            plottingF[k] = np.dot(fSquiggle,coef[2*m+1-(N-k)])-d[k]

        # interior points
        else: 
            fSquiggle = coarseF[k-int(p/2)+1:k+int(p/2)+1]
            d[k] = np.dot(fSquiggle,coef[m,:]) - refinedF[2*k+1]
            if abs(d[k]) <= eps:
                d[k] =0
            plottingF[k] = np.dot(fSquiggle,coef[m,:]) - d[k]

    return d, plottingF