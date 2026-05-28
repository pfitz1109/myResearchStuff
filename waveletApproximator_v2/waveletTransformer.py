""" WAVELET TRANSFORMER PROGRAM """

"""
    Converts a 1D signal f(x) and its 'a'-th derivative into a wavelet
    approximation. 
"""

""" LIBRARY INSTALLATION """
import numpy as np

from denseWaveletApproximation import _dense_wavelet_approximation_function
from utilities import errorEvaluation, confirmThreshold
from plottingUtilities import approximationPlot, errorDomainPlot, errorConvergencePlot

""" USER INPUT PARAMETERS """
# maximum resolution level
J = 10

# thresholding value
epsilonArray = np.logspace(-1,-9,9)

# wavelet basis order
p = 4

# signal to be approximated
def func(X):
    return 10*np.tanh(-3*X) + 10*np.tanh(-4*(X-3)) + 20

# if known - signal's 'a'-th derivative (using first derivative for simplicity)
def dFunc(X):
    return np.cos(X)

# bounds
left_bound = -10
right_bound = 10

""" Using a for-loop to move through epsilon values and see convergence """
maxErrorArray = []
for eps in epsilonArray:

    """ TRANSFORM SIGNAL """
    F, B, s0, coefficientThresholdArray, completeCoefficientArray, fApproximate = _dense_wavelet_approximation_function(p, eps, J, func, left_bound, right_bound)

    """ ERROR OF APPROXIMATION """
    errorArray, maxError = errorEvaluation(J,p,left_bound,right_bound,func,fApproximate)
    maxErrorArray.append(maxError)
    thresholdAchieved = confirmThreshold(maxError, eps)


""" PLOTTING """
# plots the last epsilon value, just for demonstration purposes

if thresholdAchieved == True:
    approximationPlot(p, J, eps, func, left_bound, right_bound, fApproximate)
    errorDomainPlot(p, J, eps, left_bound, right_bound, errorArray)

errorConvergencePlot(p, J, epsilonArray, maxErrorArray)