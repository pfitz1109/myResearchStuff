""" GENERAL PRINT DEBUGGER """
"""
    Using this space to make sure that certain parts of the code function before
    implementing them.
"""

import numpy as np
from utilities import _compute_filter_coefficients
from transformUtilities import hTildeMatrixConstructor, gTildeMatrixConstructor, thresholdCoefficients, gMatrixConstructor, hMatrixConstructor
from FWT import FWT
from BWT import BWT

np.set_printoptions(threshold=np.inf, linewidth=2000)

left_bound = 0
right_bound = 2*np.pi
p = 4
J = 7
eps = 0.00001

m = int((p-2)/2)

def func(X):
    return np.sin(X)

finestX = np.linspace(left_bound,right_bound,2**(J)*p+1)
f = func(finestX)

filterCoefficients = _compute_filter_coefficients(p)

F, d = FWT(p, J, func, left_bound, right_bound)

s0, dThreshold, dComplete = thresholdCoefficients(p,d,eps)

B, fB = BWT(p,J,dComplete)

print(abs(max(fB-f)))