""" WAVELET TRANSFORMER PROGRAM """

"""
    Converts a 1D signal f(x) and its 'a'-th derivative into a wavelet
    approximation. 
"""

""" LIBRARY INSTALLATION """
import numpy as np

from denseWaveletApproximation import _dense_wavelet_approximation_function
from utilities import errorEvaluation, confirmSignalThreshold, confirmDerivativeThreshold
from plottingUtilities import approximationPlot, errorDomainPlot, errorConvergencePlot
from derivativeUtilities import _derivative_operator_constructor

""" USER INPUT PARAMETERS """
# maximum resolution level
J = 10

# thresholding value
epsilonArray = np.logspace(-1,-9,9)

# wavelet basis order
p = 6

# spatial derivative to be approximated
a = 2

# signal to be approximated
def func(X):
    return np.exp(-4*X**2)

# if known - signal's 'a'-th derivative (using first derivative for simplicity)
def dFunc(X):
    return -8*np.exp(-4*X**2) + 64*np.exp(-4*X**2)*X**2

# bounds
left_bound = -5
right_bound = 5

# status_update - if set to true, will print out status updates
status_update = False

""" Using a for-loop to move through epsilon values and observe convergence """
maxErrorArray = []; maxDerivativeErrorArray=[]
for eps in epsilonArray:

    """ TRANSFORM SIGNAL """
    F, B, s0, d, coefficientThresholdArray, completeCoefficientArray, fApproximate = _dense_wavelet_approximation_function(p, eps, J, func, left_bound, right_bound)

    """ GENERATE DERIVATIVE OPERATORS """
    # entry needs to be J-1, I messed up somewhere in my construction that the
    # J used in the FWT and BWT must be one value greater than the entry used
    # in the construction of the Gamma operator
    derivativeOperator = _derivative_operator_constructor(p,a,left_bound,right_bound,J-1,F,B,status_update)

    """ APPROXIMATE DERIVATIVE """
    derivativeCoefficients = derivativeOperator @ completeCoefficientArray 
    derivativeApproximate = B @ derivativeCoefficients

    """ ERROR OF SIGNAL APPROXIMATION """
    errorArray, maxError = errorEvaluation(J,p,left_bound,right_bound,func,fApproximate)
    maxErrorArray.append(maxError)
    thresholdAchieved = confirmSignalThreshold(maxError, eps)

    """ ERROR OF DERIVATIVE APPROXIMATION """
    derivativeErrorArray, maxDerivativeError = errorEvaluation(J,p,left_bound,right_bound,dFunc,derivativeApproximate)
    maxDerivativeErrorArray.append(maxDerivativeError)
    derivativeThresholdAchieve = confirmDerivativeThreshold(maxDerivativeError, eps, p, a)


""" PLOTTING """

# plots approximation and error on domain using last epsilon value
# just for demonstration purposes
if thresholdAchieved == True:
    approximationPlot(p, J, eps, func, left_bound, right_bound, fApproximate)
    errorDomainPlot(p, J, eps, left_bound, right_bound, errorArray)
    approximationPlot(p,J,eps,dFunc,left_bound,right_bound,derivativeApproximate)
    errorDomainPlot(p,J,eps,left_bound,right_bound,derivativeErrorArray)

# plots error convergence for each epsilon value tested
errorConvergencePlot(p, J, epsilonArray, maxErrorArray)
errorConvergencePlot(p, J, epsilonArray, maxDerivativeErrorArray)