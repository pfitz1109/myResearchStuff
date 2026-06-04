""" DERIVATIVE UTILITIES """

"""
    Utilities necessary for constructing the derivative operators.
"""

import numpy as np
import time
from scipy.linalg import null_space, toeplitz
from scipy.special import factorial 
from utilities import _compute_filter_coefficients, confirmContinuity, _lagrange_boundary_values


# constructs the 'chi' eigenvector that is necessary for the interior rows of 
# the gamma operator. see pp. 12 in "How to Wavelet" for equations
def _chi_constructor(p,a,left_bound, right_bound, status_update=False):
    if status_update:
        start_time = time.time()
        print('Solving eigenvalue problem...')
    # run a check to make sure that sufficient differentiability is provided
    confirmContinuity(p,a)

    # determine number of boundary condition rows
    m = int((p-2)/2)

    # grab the filter coefficients corresponding to the interior of the domain
    filterCoefficients = _compute_filter_coefficients(p)
    interiorCoefficients = filterCoefficients[m,:]

    # construct the h-vector - NOT the same as above! has zeros spliced in 
    # it and an entry of unity at the center
    h = np.zeros(2* len(interiorCoefficients)+1, dtype=interiorCoefficients.dtype)
    h[1::2] = interiorCoefficients

    mid_index = len(h)//2
    h[mid_index] = 1

    # h now looks like [0 h0 0 h1 0 ... 0 h_(p/2-1) 1 h_(p/2+1) 0 ... 0 h_(p-1) 0 h_p 0]

    # construct the HMatrix
    HMatrix = np.zeros((2*p+1,2*p+1))

    # going to loop through each point and construct the matrix - seems easiest
    # way to do it 
    for i in range(0,2*p+1):
        for j in range(0,2*p+1):
            for mu in range (0,2*p):
                if (j==(2*i-mu)): # kroenecker delta condition in Eq. 19 of HtW
                    HMatrix[i,j] = h[mu]
            
    # we know a priori that the eigenvalue associated with the eigenvector
    # we are looking for should be (1/2)^n, so we need to solve the specific 
    # eigenproblem 
    #               H . xi = (1/2)^n . xi
    # can't just find the eigenvalues and eigenvectors of H - we need THIS 
    # specific eigenvector
    eigenvalue = (1/2)**(a)

    # generate the null-space problem
    A = HMatrix - eigenvalue * np.eye(HMatrix.shape[0])

    # with eigenvalues, can now solve the Null-Space problem to compute the 
    # corresponding eigenvector
    # note that this produces an eigenvector with norm = 1
    eigenvector = null_space(A)

    # grab the grid spacing on the coarsest level - needed for normalization
    dX = (right_bound - left_bound)/(2*p)

    # right hand side of the normalization equation 
    rhsTerm = factorial(a)*(-1 / dX)**a

    # left hand side of the normalization equation - can be expressed as a dot
    # product between (i)^m for i:[0,2p] and the eigenvector computed above
    vec = (np.arange(0,2*p+1))**(a)
    lhsTerm = np.dot(vec,eigenvector)

    # the normalization factor is equal to the RHS divided by the LHS - this is
    # pulled straight from the Mathematica code
    normFactor = rhsTerm/lhsTerm

    # now we multiply the eigenvector by this norm factor to obtain chi
    chi = eigenvector*normFactor
    
    # for sake of eliminating round-off error, going to threshold any entry
    # smaller than 1e-10 to zero. this condition comes from when a=4,
    # p=10 - anything larger than this, we consider significant
    # run the condition a=4, p=10 and you'll see that entries that should be 
    # zero are written as like 3 e-12
    chi[abs(chi) < 1e-10] = 0

    if status_update:
        end_time = time.time()
        print('Eigenvalue problem solved.')
        print(f'Time to solve eigenvalue problem: {end_time-start_time:.3f}')
    
    return np.array(chi).squeeze()

# pp.13 in H2W makes no sense so I had to reference the Mathematica MRWT 
# toolbox for this
# similar to the 'border' function in the Mathematica MRWT toolkit. Will
# produce a matrix used to construct the boundary weights 
def _weight_matrix_constructor(p,k):
    if (k > p+1):
        raise ValueError('Error in constructing Gamma Weight Matrix. '
                         f'Index k={k} is out of bounds for p={p}. '
                         'Hint: k must be less than or equal to p+1. ')
    # need the eigenvector for computation (eq. 24)

    # generate an empty vector that we will populate with weights 
    weightVector = np.zeros(2*p+1)

    # entry of unity at the following index
    weightVector[(p+1)-k] = 1

    # fill in values of where the lagrange boundary function should apply
    for j in range(p+1,2*p-1):
        weightVector[j] = _lagrange_boundary_values( k-p//2, p, 2-p+1-p//2+2*p-2-j)

    # using a toeplitz construction - Gemini recommended this
    first_col = np.zeros(len(weightVector))
    weightMatrix = toeplitz(first_col, weightVector)

    return weightMatrix

# compute the weights necessary for Gamma and store them in a matrix
# similar to the 'calcA' function in the Mathematica MRWT toolbox 
def _gamma_weight_constructor(p,a,left_bound,right_bound,status_update=False):
    # first compute the eigenvector chi, this will be necessary
    chi = _chi_constructor(p,a,left_bound,right_bound,status_update)

    # next build an empty matrix that we will store the weights 
    gammaCoefficients = np.zeros((p+1,2*p+1))

    # will assign the last row equal to chi
    gammaCoefficients[-1,:] = chi

    # now have to iterate through each row and perform matrix multiplication
    # to determine weights 
    for r in range(1,p+1):
        M = _weight_matrix_constructor(p,r)
        gammaCoefficients[r-1,:] = M @ chi
    
    return gammaCoefficients

# construct the gamma operator matrix 
def _gamma_constructor(p,a,left_bound,right_bound,J,status_update=False):
    if status_update:
        print('Generating Gamma operator...')
    # need to generate the weights for this matrix
    gammaCoefficients = _gamma_weight_constructor(p,a,left_bound,right_bound,status_update)

    # grab the interior weights 
    interiorCoefficients = gammaCoefficients[-1,:]

    # grab the boundary weights
    boundaryCoefficients = gammaCoefficients[:p,:]

    # these variable declarations will be useful later on
    N = 2**(J+1)*p+1
    K = gammaCoefficients.shape[1]

    # generate a blank matrix that we'll fill in
    Gamma = np.zeros((N,N))

    """ LEFT-BOUNDARY TERMS """
    # insert the top p-rows in the gammaCoefficients matrix into the top-left
    # corner of the Gamma matrix
    Gamma[:p, :K] = boundaryCoefficients

    """ INTERIOR TERMS """
    # beginning at row r=p, we start filling in entries and slide the 
    # interiorCoefficients by one column with each row 
    for r in range(p, N-p):
        Gamma[r, (r-p):(r-p+K)] = interiorCoefficients

    """ RIGHT-BOUNDARY TERMS """
    # we have to take the left-boundary 'matrix' and 'flip' it along the 
    # horizontal and vertical axes. then we multiply by (-1)^a, that is,
    # odd derivatives will have negative entries here but positive orders 
    # will not 
    Gamma[-p:, -K:] = (-1)**(a)*np.flip(boundaryCoefficients)
    
    # have to multply by the weighting factor 2**(J*a)
    Gamma = 2**(a*J)*Gamma
    
    return Gamma

# construct the derivative operator matrix D
# note that F and B should be built to highest resolution size
def _derivative_operator_concstructor(p,a,left_bound,right_bound,J,F,B,status_update=False):

    if status_update:
        print('Constructing derivative operator...')

    Gamma = _gamma_constructor(p,a,left_bound,right_bound,J,status_update)

    # for some reason you need to transpose Gamma? I only saw this in the 
    # Mathematica toolbox
    derivativeOperator = F @ np.transpose(Gamma) @ B

    if status_update:
        print('Derivative operator built.')

    return derivativeOperator
