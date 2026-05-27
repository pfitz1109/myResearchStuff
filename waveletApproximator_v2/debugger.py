""" GENERAL PRINT DEBUGGER """
"""
    Using this space to make sure that certain parts of the code function before
    implementing them.
"""

import numpy as np
from utilities import _compute_filter_coefficients, hTildeMatrixConstructor, gTildeMatrixConstructor
from FWT import FWT

left_bound = 0
right_bound = 2*np.pi
p = 4
J = 3
eps = 0.01

m = int((p-2)/2)

def func(X):
    return np.sin(X)

d = []
finestX = np.linspace(left_bound,right_bound,2**(J)*p+1)
f = func(finestX)

F, d = FWT(p, 3, func, left_bound, right_bound)
print(d)