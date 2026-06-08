""" LAGRANGE BOUNDARY VALUE TEST """

"""
    Tests to make sure that the Lagrange boundary matrix function found in
    utilities.py produces the correct results. 

    Correct results were generated using the 'border' function found in the
    Mathematica MRWT toolbox. 
"""

import unittest
import numpy as np
from utilities import _lagrange_boundary_values

