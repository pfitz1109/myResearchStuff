""" MASTER EXECUTABLE FILE """
""" 
    Main goal with this file was to try and incorporate the functional wavelet coefficient computations together with a derivative
    file, since the functional values are required for the computation of the derivative coefficients. I realize writing this code now that 
    I did not write the "denseFunctionalWaveletCoefficient" file in a way that it can be called as a function. 
    Additionally, the derivative operation is slightly more complicated than linear algebra. You would need to construct the forward and backward wavelet transforms.
    
    User-defined inputs:
    basis : basis orders to simulate
    threshold : epsilon values to simulate
    func : function to approximate
    left_bound : left-bound of the domain
    right_bound : right-bound of the domain

    Main goals of this code:
    1. Evaluate the sparse wavelet coefficients for various combinations of p and epsilon
    3. Plot the maximum absolute error 
"""

""" LIBRARY INSTALLATION """
import numpy as np
import matplotlib.pyplot as plt
from plottingUtilities import waveletApproximationPlotting, waveletApproximationErrorPlotting
from FWT import FWT
from BWT import BWT
from denseFunctionalWaveletCoefficient import denseFunctionalWaveletCoefficient

""" USER DEFINED INPUT PARAMETERS """
basis = [4, 6, 8, 10]
threshold = np.logspace(-1,-10,10)

def func(X): 
    return 10*np.tanh(-3*X) + 10*np.tanh(-4*(X-3)) + 20

left_bound = -10
right_bound = 10

""" SIMULATION LOOP """
"""   
    For each interpolation basis 'p' in above array:
        Run the denseFunctionalWaveletCoefficient function for each of the given epsilon values

"""
for p in basis:

    maxError = []

    for thresh in threshold:
        approximateX, approximateF, errorX, errorF, absoluteError, j_final = denseFunctionalWaveletCoefficient(thresh, p, 10, func, left_bound, right_bound)

        # plot the wavelet approximation against the analytical solution 
        waveletApproximationPlotting(func, approximateX, approximateF, thresh, left_bound, right_bound)

        # plot the error across the domain for the final resolution level achieved
        # waveletApproximationErrorPlotting(errorX, absoluteError, j_final, thresh)

        # we really only want the maximum of the absolute error to generate the convergence plot 
        maximumError = max(abs(absoluteError))
        maxError.append(maximumError)
        

    maxError = np.array(maxError)

    # Scatter plot
    plt.scatter(threshold, maxError)

    # Linear fit in log-log space
    coeffs = np.polyfit(np.log10(threshold),
                        np.log10(maxError), 1)

    slope = coeffs[0]
    intercept = coeffs[1]

    # Trend line
    fitLine = 10**intercept * threshold**slope

    plt.plot(
        threshold,
        fitLine,
        linestyle='--',
        label=f'p={p}: {slope:.3f}'
    )

    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$\|\cdot\|_\infty$')
    plt.title(f'Max Absolute Error vs. Threshold for p={p} Interpolation Order')

    plt.xscale('log')
    plt.yscale('log')

    plt.legend()

    plt.show()