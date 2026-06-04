""" Utility functions for sparseWaveletApproximator.py and denseWaveletApproximator.py """
import numpy as np

# validates p, checks that it is a positive even integer less than or equal to 10
def _validate_p(p: int) -> None:
    if p <=0 :
        raise ValueError(f'p must be a positive integer, got p={p}')
    if p % 2 != 0:
        raise ValueError(f'p must be an even integer, got p={p}')
    if p > 10:
        raise ValueError(f'Code can only handle interpolation orders up to p=10, please try a smaller interpolation order. Got p={p}')

# validates eps, checks that it is a positive float 
def _validate_eps(eps: float) -> None: 
    if eps <= 0:
        raise ValueError(f'Thresholding value "eps" must be greater than zero, got {eps}')

# neville's theorem for computing the filter coefficients h
def _compute_filter_coefficients(p: int) -> np.ndarray:
    coef = np.ones((p-1,p))

    for i in range (0,p-1):
        for j in range(0,p):
            for k in range(0,p):
                if (k==j):
                    continue
                coef[i,j] = coef[i,j]*(i+0.5-k)/(j-k)
    return coef

# returns the absolute error at each collocation point and the maximum error
# for a given approximation against the exact solution
def errorEvaluation(J,p,left_bound, right_bound, func, fApproximate):
    # generate fExact on grid using  2**(J)*p+1 points
    xExact = np.linspace(left_bound, right_bound, 2**(J)*p+1)
    fExact = func(xExact)

    # compute error array on domain
    errorArray = abs(fExact - fApproximate)
    maxError = max(errorArray)
    print(f'Maximum Absolute Error on Domain: {maxError}')

    return errorArray, maxError

# compares the maximum absolute error to the thresholding value 
def confirmSignalThreshold(maxError, eps):
    # should be on the same order of magnitude - does not need to be exactly
    # the same 
    result = maxError < eps*10
    if result == False:
        print('Maximum error on domain is greater than epsilon. Cannot guarantee accuracy of approximation.')
    else:
        print('Maximum error on domain is on the order of magnitude of epsilon. Accuracy guaranteed.')
    return result

# compares the maximum absolute error for a derivative to the thresholding value
# note that the guaranteed order of accuracy is different than previous function
def confirmDerivativeThreshold(maxError, eps, p, a):

    result = abs(maxError) < (eps**(1-a/p))*10

    if result == False:
        print('Maximum derivative error on domain has higher order of magnitude than epsilon^(1-a/p). Cannot guarantee accuracy of approximation.')
    else:
        print('Maximum derivative error on domain is on the order of magnitude of epsilon^(1-a/p). Accuracy guaranteed.')

# ensures that proper basis is used for specified derivative computation
def confirmContinuity(p,a) -> None:
    if p == 4:
        if a > 1:
            raise ValueError('Insufficient basis order for chosen derivative. ' 
                             'For p=4, up to first derivatives can be computed. ' 
                             f'Received a={a}.')
    if p == 6 :
        if a > 2:
            raise ValueError('Insufficient basis order for chosen derivative. '
                             'For p=6, up to second derivatives can be computed. '
                             f'Received a={a}.')
    if p == 8 :
        if a > 3:
            raise ValueError('Insufficient basis order for chosen derivative. ' 
                             'For p=8, up to third derivatives can be computed. ' 
                             f'Received a={a}.')
    if p == 10 :
        if a > 4 :
            raise ValueError('Insufficient basis order for chosen derivative. ' 
                             'For p=10, up to fourth derivatives can be computed. ' 
                             f'Received a={a}.')

# lagrange boundary functions - necessary for _gamma_boundary function in 
# transformUtilities.py
def _lagrange_boundary_values(m: int,p: int,x: int) -> float:
    value = 1
    # // indicates integer division 
    for n in range(1-p//2,p//2+1):
        if n==m:
            continue
        value = value*(x-n)/(m-n)

    return value