"""" BACKWARD WAVELET TRANSFORM FUNCTION """

""" 
    This function generates the 'B' matrix that converts discrete wavelet points
    to physical values. It makes use of the filter coefficient function 
    defined in the "utilities" program to construct the entries within 
    the matrix, as well as the hMatrix and gMatrix constructors in the 
    "transformUtilities" program. 
    Note that the matrix is not constructed sparsely for simplicity - this is NOT
    meant to be a computationally-efficient program, but instead one that achieves
    the correct results. 
"""