""" GENERAL PRINT DEBUGGER """
"""
    Using this space to make sure that certain parts of the code function before
    implementing them.
"""

import numpy as np
from utilities import _compute_filter_coefficients
from transformUtilities import hTildeMatrixConstructor, gTildeMatrixConstructor, thresholdCoefficients, gMatrixConstructor
from FWT import FWT

left_bound = 0
right_bound = 2*np.pi
p = 4
J = 3
eps = 0.01

m = int((p-2)/2)

def func(X):
    return np.sin(X)

finestX = np.linspace(left_bound,right_bound,2**(J)*p+1)
f = func(finestX)

F, d = FWT(p, J, func, left_bound, right_bound)

s0, dThreshold, dComplete = thresholdCoefficients(p,d,eps)

fApproximate = np.linalg.solve(F,dComplete)

print(gMatrixConstructor(p,1))
