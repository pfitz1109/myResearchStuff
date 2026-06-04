""" GENERAL PRINT DEBUGGER """
"""
    Using this space to make sure that certain parts of the code function before
    implementing them.
"""

import numpy as np
from utilities import _compute_filter_coefficients, _lagrange_boundary_values, errorEvaluation, confirmSignalThreshold, confirmDerivativeThreshold
from transformUtilities import hTildeMatrixConstructor, gTildeMatrixConstructor, thresholdCoefficients, gMatrixConstructor, hMatrixConstructor
from FWT import FWT
from BWT import BWT
from denseWaveletApproximation import _dense_wavelet_approximation_function
from derivativeUtilities import _gamma_constructor, _derivative_operator_concstructor
from plottingUtilities import approximationPlot, errorDomainPlot

np.set_printoptions(threshold=np.inf, linewidth=2000)

left_bound = 0
right_bound = 2*np.pi
p = 6
fwtJ = 8
gammaJ = fwtJ - 1 
eps = 0.001
a = 2
status_updates = False

def func(X):
    return np.sin(X)

def derivative(X):
    return -np.sin(X)

xFinest = np.linspace(left_bound, right_bound, 2**(fwtJ+1)*p+1)
fExact = func(xFinest)
firstDerivativeExact = derivative(xFinest)

F, d = FWT(p,fwtJ,func,left_bound,right_bound,status_updates)
s0, coefficientThreshold, dComplete = thresholdCoefficients(p,d,eps,status_updates)
B, fApproximate = BWT(p,fwtJ,dComplete,status_updates)

Gamma = _gamma_constructor(p,a,left_bound,right_bound,gammaJ,status_updates)

derivativeOperator = _derivative_operator_concstructor(F,B,Gamma,status_updates)

firstDerivativeCoefficients = derivativeOperator @ dComplete
firstDerivativeApproxiamtes = B @ firstDerivativeCoefficients

approximationPlot(p,fwtJ,eps,func,left_bound,right_bound,fApproximate)
approximationPlot(p,fwtJ,eps,derivative,left_bound,right_bound,firstDerivativeApproxiamtes)

firstDerivativeErrorArray, maxFirstDerivativeError = errorEvaluation(fwtJ,p,left_bound,right_bound,derivative,firstDerivativeApproxiamtes)
thresholdAchieved = confirmDerivativeThreshold(maxFirstDerivativeError,eps,p,a)

errorDomainPlot(p,fwtJ,eps,left_bound,right_bound,firstDerivativeErrorArray)