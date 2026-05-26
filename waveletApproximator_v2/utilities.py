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