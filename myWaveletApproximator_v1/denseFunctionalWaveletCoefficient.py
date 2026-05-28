""" Dense wavelet approximator """
""" There will be a separate file that does the sparse computations, for now I just wanted to get the general idea of wavelet transformations down """

""" P.S. FitzGerald, 5/12/26 """

""" CODING NOTES """

""" 
Inputs: eps (threshold), p (interpolation basis), J (maximum resolution level), func (function to be appoximated),
left_bound (left boundary of domain), right_bound (right boundary of domain)
Outputs: sparse x-coordinates, sparse function approximation values, x-grid for computing the error, funtion approximations for 
        computing the error, absolute error of approximation to analytical solution on x-grid for computing error, final resolution level 
        achieved
Functiona Goals: 
    1. Computes the wavelet coefficients on a dense grid. A pseudo-sparse computation is performed to see what level of resolution
        is necessary to achieve the desired threshold. Either the maximum level J is reached, or all the wavelet coefficients are below
        the threshold. Will terminate computations once either of these conditions are met. 
    2. Plots the grid points by resolution level across the domain for a given threshold value (if called).
"""

import numpy as np
import matplotlib.pyplot as plt
from utilities import _validate_p, _validate_eps, _compute_filter_coefficients, wavelet_coefficient_generator # type: ignore
from plottingUtilities import waveletResolutionLevelsPlotting

def denseFunctionalWaveletCoefficient(eps,p, J, func, left_bound, right_bound):    

    # compute the number of boundary "terms" for a given basis order
    m = int((p-2)/2)

    # validates that p is a positive, even integer less than or equal to 10, and that eps is positive
    _validate_p(p)
    _validate_eps(eps)

    """ S0 COMPUTATIONS """
    # define coarsest grid 
    X0 = np.linspace(left_bound, right_bound, 2*p+1)

    # "s" coefficients 
    s0 = func(X0)

    """ Neville's Theorem for computing the filter coefficients h """
    """ 
        there are m=(p-2)/2 boundary points for a given basis order. The filter coefficent matrix "coef" is generated in an 
        array format like so :

        rows 0-(m-1): left-boundary filter coefficients (row 0: leftmost, row1: second from left, etc.)
        row m: interior filter coefficients
        rows (m+1)-p: right-boundary filter coefficients (row p: rightmost, row(p-1): second from right, etc.)
    """

    coef = _compute_filter_coefficients(p)

    # generate empty lists that we will apend x-values and approximate function values for plot of approximation
    nodeLocations = []
    approximateValues = []\
    
    # generate empty lists that we will append x-values and resolution levels for plot of resolution across domain
    xLevelsPlotting = []
    levelsPlotting = []

    # append the s0 coefficients and their corresponding x-coordinates
    nodeLocations.append(X0); approximateValues.append(s0)

    """ THRESHOLDING COEFFICIENTS """   

    # have to redefine the coarse grid here for the for-loop
    coarseX = X0

    # begin for-loop to compute d-coefficients at each collocation point
    for j in range(1,J+1):

        # comnpute dx_j to get an idea of the resolution we are using
        dx = (right_bound-left_bound)/(2**(j+1)*p)

        # evaluate function on previous mesh, necessary for fSquiggle vector used to compute thresholding coefficients
        coarseF = func(coarseX)

        # define new mesh 
        refinedX = np.linspace(left_bound, right_bound, 2**(j+1)*p+1)

        # evaluate function on the new mesh 
        refinedF = func(refinedX)

        # initialize this array to store the function values we will plot later
        plottingF = np.zeros(2**j*p)

        """ This is where the magic happens - computes the coefficients at the current resolution level"""
        d, plottingF = wavelet_coefficient_generator(p, j, m, eps, coef, coarseF, refinedF)

        # grab the index of nonzero coefficients for plotting purposes
        plottingIndicies = np.where(d !=0)[0] 

        # generate x-grid for current resolution level - does NOT include points on previous grids, hence the use of np.arange
        xPlotting = np.arange(left_bound+dx, right_bound, 2*dx)

        # establish termination conditions 

        # first check to see if we have reached our user-defined maximum resolution level
        if (j==J):
            print(f'\nReached maximum resolution level J={J} without all wavelet coefficients falling below thresholding level.')
            break
        # then check if all the coefficients are less than our threshold
        elif (np.count_nonzero(d) == 0):
            print(f"Maximum Resolution Required for {eps} Accuracy and Basis Order p = {p}: j = {j-1}")
            break

        # if we have not reached termination conditions, continue with the below operations

        # save the x-coordinates with nonzero wavelet coefficients
        nodeLocations.append(xPlotting[plottingIndicies]) # save the x-coordinates of the non-zero coefficients
        approximateValues.append(plottingF[plottingIndicies]) # save the approximate function values corresponding to the above x-coordinates
        
        # after thresholding, check to see if the coefficient magnitude is greater than zero. if yes (which is a 1), multiply by j. this returns the 
        # resolution level at the corresponding collocation point. Probably a more efficient way to do this, but this is easiest.
        sparsePlotting = np.float64(abs(d) > 0)*j

        # for those coefficients that are zero, set them equal to NaN - Python ignores them while plotting
        sparsePlotting[sparsePlotting == 0] = np.nan

        # storing resolution level and x-coordinate for plotting
        xLevelsPlotting.append(xPlotting)
        levelsPlotting.append(sparsePlotting)

        # redefine new mesh as old mesh - need this for next resolution level
        coarseX = refinedX

    """ 
        Note that the way this code is written, the above for-loop will generate the coefficients for the resolution level ABOVE what is necessary for 
        a certain threshold. For example, if only j=1 resolution is required to achieve a threshold of eps=0.01, the for-loop will set and check the resolution
        level j=2. The coefficients, function values, and plotting information is computed and stored for this next-highest resolution level. This makes 
        error computation super simple.
    """

    # convert lists of significant x-coordinates and function approximations into numpy arrays
    approximateX = np.concatenate(nodeLocations); approximateF = np.concatenate(approximateValues)

    """ ERROR APPROXIMATION """
    """ To compute the error of the approximation, we step to the next resolution level, compute the dense wavelet approximation, and then generate an approximation
        of the function. We then compare this to the analytical function evaluated on that grid.
        Because we have already generated this next-highest level in the for-loop (see above comment block), we can pull that information and use it. We don't have to 
        go through another computation of it.
    """

    # generate grid for j_final + 1 resolution level and evaluate the analytical function on this grid
    errorX = np.arange(left_bound+dx, right_bound, 2*dx)
    errorF = func(errorX)

    # plotting F comes from for-loop 
    absoluteError = abs(errorF - plottingF)

    """ plot the resolution level at a given collocation point """
    waveletResolutionLevelsPlotting(X0, xLevelsPlotting, levelsPlotting, eps)

    """ Function Outputs: 
        sparse x-coordinates
        sparse function approximation values
        x-grid for computing the error
        funtion approximations for computing the error
        absolute error of approximation to analytical solution on x-grid for computing error
        final resolution level achieved 
    """
    return (approximateX, approximateF, errorX, errorF, absoluteError, j)