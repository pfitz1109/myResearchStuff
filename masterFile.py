""" MASTER EXECUTABLE FILE """
""" 
    Main goal with this file was to try and incorporate the functional wavelet coefficient computations together with a derivative
    file, since the functional values are required for the computation of the derivative coefficients. I realize writing this code now that 
    I did not write the "denseFunctionalWaveletCoefficient" file in a way that it can be called as a function. 
    Additionally, the derivative operation is slightly more complicated than linear algebra. You would need to construct the forward and backward wavelet transforms.
    
"""
""" LIBRARY INSTALLATION """
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from utilities import _validate_p, _validate_eps, _compute_filter_coefficients # type: ignore
from FWT import FWT
from BWT import BWT
from denseFunctionalWaveletCoefficient import denseFunctionalWaveletCoefficient

""" USER DEFINED INPUT PARAMETERS """
# interpolation order, acceptable error, maximum resolution
# accepts only values of p less than or equal to 10
""" p = 6
_validate_p(p)
# define number of boundary conditions on each side
m = int((p-2)/2)
eps = 1e-6
_validate_eps(eps)
J = 10 """

basis = [4, 6, 8, 10]
threshold = np.logspace(-1,-10,10)

for p in basis:

    maxError = []

    for thresh in threshold:
        error = denseFunctionalWaveletCoefficient(thresh, p)
        maxError.append(error)

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
plt.title('Max Absolute Error vs. Threshold')

plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.grid(True, which='both')

plt.show()