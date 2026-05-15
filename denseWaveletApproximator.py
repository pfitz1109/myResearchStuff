""" Dense wavelet approximator """
""" There will be a separate file that does the sparse computations, for now I just wanted to get the general idea of wavelet transformations down """

""" P.S. FitzGerald, 5/12/26 """

""" CODING NOTES """

""" 
    1. Noticing that some of the wavelet coefficients reach machine precision so the computer overwrites them as zeros. Explains some of the "gaps" you might see
    at lower resolution levels. Shouldn't technically be there, but the coefficients are so small anyways it doesn't really matter.
    2. In general - for a centrally-interesting function (i.e., sharp increase and decrease at the center of the function), the wavelet resolution level should 
    look like a pyramid. 
    3. Adding this commen just to see how Git and GitHub responds. Woohoo!
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
from utilities import _validate_p, _validate_eps, _compute_filter_coefficients # type: ignore

# clear the output
subprocess.run('clear', shell=True)    


""" USER INPUT PARAMETERS """
# interpolation order, acceptable error, maximum resolution
# accepts only values of p less than or equal to 10
p = 6
_validate_p(p)
eps = 1e-6
_validate_eps(eps)
J = 10

# domain
left_bound = -3*np.pi
right_bound = 3*np.pi

# function to be approximated
def func(X):
    f = 10*np.tanh(5*-X)+10
    return f

""" S0 COMPUTATIONS """
# coarsest grid 
X0 = np.linspace(left_bound, right_bound, 2*p+1)

# "s" coefficients 
s0 = func(X0)

""" Neville's Theorem for computing the filter coefficients h """
coef = _compute_filter_coefficients(p)
# number of boundary conditions
m = int((p-2)/2)

""" THRESHOLDING COEFFICIENTS """
# generate empty lists that we will apend x-locations and approximate function values
nodeLocations = []
approximateValues = []

coarseX = X0
# sticking this here so this appears as the first array to be plotted 
plt.scatter(X0, np.zeros(len(X0)), label='Resolution Level j = 0')
# for-loop that computes the thresholding coefficients for a given resolution level
# d = dot(filterCoefficients,coarseF[neighboringPoints]) - refinedF[currentNode] 
for j in range(1,J+1):
    print(f'\n###### Resolution Level {j} #######')
    # number of midpoint nodes generated
    N = 2**(j)*p
    # generate d vector, will store thresholding coefficients in here
    d = np.zeros(N)
    # evaluate function on previous mesh, necessary for fSquiggle vector used to compute thresholding coefficients
    coarseF = func(coarseX)

    # define new mesh 
    refinedX = np.linspace(left_bound, right_bound, 2**(j+1)*p+1)
    # comnpute grid step size to get an idea of the resolution we are using
    gridStepSize = (right_bound-left_bound)/(2**(j+1)*p)
    print(f'Grid Step Size: {gridStepSize:.3f}')

    # evaluate function on the new mesh, needed for thresholding computation 
    refinedF = func(refinedX) 
    plottingF = np.zeros(N)
    # going through every midpoint in the refined grid
    for k in range(0,N):
        # at each node, compute the corresponding fSquiggle vector; compute d at the node; compute the approximated (exact?) function value
        # left boundary condition(s)
        if k < m:
            fSquiggle = coarseF[0:p]
            d[k] = np.dot(fSquiggle,coef[k,:]) - refinedF[2*k+1]
            plottingF[k] = np.dot(fSquiggle,coef[k,:]) - d[k]
        # right boundary condition(s)
        elif k > N-(m+1):
            fSquiggle = coarseF[-p:]
            d[k] = np.dot(fSquiggle,coef[2*m+1-(N-k),:]) - refinedF[2*k+1]
            plottingF[k] = np.dot(fSquiggle,coef[2*m+1-(N-k)])-d[k]
        # interior points
        else: 
            fSquiggle = coarseF[k-int(p/2)+1:k+int(p/2)+1]
            d[k] = np.dot(fSquiggle,coef[m,:]) - refinedF[2*k+1]
            plottingF[k] = np.dot(fSquiggle,coef[m,:]) - d[k]

    # thresholding operation
    dThreshold = np.copy(d)
    dThreshold[abs(dThreshold) <= eps] = 0
    plottingIndicies = np.where(dThreshold !=0)[0] # grabs the indices of the nonzero (non-thresholded) nodes for plotting purposes
    # generate plotting nodes for this specfic 
    xPlotting = np.arange(left_bound+gridStepSize, right_bound, 2*gridStepSize)

    # establish termination conditions 
    if (j==J):
        print(f'\nReached maximum resolution level J={J} without all wavelet coefficients falling below thresholding level. Final wavelet coefficients for resolution level j={j} with grid step size {gridStepSize:.3f}:')
        print(d)
    elif (np.count_nonzero(dThreshold) == 0):
        print('\nAll wavelet coefficients are below thresholding level, terminating algorithm.')
        print(f'Final wavelet coefficients for resolution level j={j} with grid step size {gridStepSize:.3f}:')
        print(d)
        break

    # for plotting of the wavelet approximation 
    nodeLocations.append(xPlotting[plottingIndicies]) # save the x-coordinates with nonzero wavelet coefficients
    approximateValues.append(plottingF[plottingIndicies]) # save the approximate function values corresponding to the above x-coordinates
    
    # storing data for grid resolution figure
    sparsePlotting = np.float64(abs(dThreshold) > 0)*j
    sparsePlotting[sparsePlotting == 0] = np.nan
    plt.scatter(xPlotting, sparsePlotting, label = f'Resolution Level j = {j}')

    # redefine new mesh as old mesh - need this for next resolution level
    coarseX = refinedX

""" PLOTTING """

""" plotting resolution and nodes across grid """
plt.xlabel('x'); plt.ylabel('Resolution Level'); plt.title('Resolution Levels Across Grid'); plt.legend(loc = 'best')
plt.show()
plottingX = xPlotting

""" wavelet reconstructed approximation """
# convert the appended lists into numpy arrays
approximateX = np.concatenate(nodeLocations); approximateF = np.concatenate(approximateValues)
plt.scatter(approximateX, approximateF, label ='Wavelet Approximations')
# plotting analytical function
functionalX = np.linspace(left_bound,right_bound,1000)
functionalY = func(functionalX)
plt.plot(functionalX,functionalY, label='Analytical Solution', color='r')
plt.title('Analytical vs Approximate'); plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend();
plt.show()