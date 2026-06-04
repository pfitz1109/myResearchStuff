""" NEVILLE'S THEOREM UNIT TEST """

"""
    Unit test that confirms the proper construction of the filter coefficient
    matrix using Neville's Theorem.
"""

import unittest
import numpy as np
from utilities import _compute_filter_coefficients

class TestNevillesTheorem(unittest.TestCase):
    def test_coefficients(self):
        # test to make sure that p=4 generates the correct coefficient matrix
       expected = [[5/16, 15/16, -5/16, 1/16], [-1/16,9/16,9/16,-1/16], [1/16, -5/16, 15/16, 5/16 ]]
       actual = _compute_filter_coefficients(4)

       np.testing.assert_allclose(actual,expected, atol=1e-7)
