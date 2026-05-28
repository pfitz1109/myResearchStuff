""" PLOTTING UTILITIES """
"""
    Various plotting functions that can be called to plot different trends
    for a given dense wavelet approximation.
"""

import numpy as np
import matplotlib.pyplot as plt

# plot the dense wavelet approximation against high-resolution analytical 
# function 
def approximationPlot(p,J, eps, func, left_bound, right_bound, fApproximate) -> None:
    print('Plotting approximation versus analytical signal...')
    # generate "precise" function vector
    xAnalytic = np.linspace(left_bound, right_bound, min(int(1/eps),10000))
    fAnalytic = func(xAnalytic)
    plt.plot(xAnalytic, fAnalytic, label = 'Analytical Solution', color='red')

    # generate xFinest
    xFinest = np.linspace(left_bound, right_bound, 2**(J)*p+1)

    # scatter plot of wavelet approximations
    plt.scatter(xFinest, fApproximate, label='Wavelet Approximation')

    # title and labels
    plt.title(fr'Dense Wavelet Approximation Using J={J} and $\varepsilon$={eps}')
    plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend()
    plt.show()

# plots absolute error across the domain
def errorDomainPlot(p, J, eps, left_bound, right_bound, errorArray) -> None:
    print('Plotting absolute error across the given domain...')

    # generate x-grid
    xFinest = np.linspace(left_bound, right_bound, 2**(J)*p+1)

    # line plot of error
    plt.plot(xFinest, errorArray, label='Absolute Error')
    plt.title(rf'Absolute Error Over Domain Using J={J} and $\varepsilon$={eps}')
    plt.xlabel('x'); plt.ylabel(r'$||\cdot||_\infty$')
    plt.show()

# plots an array of absolute maximum errors against corresponding thresholds
# used to check convergence of method
# might not be exactly correct because we are not going up to the next 
# resolution level but whatever
def errorConvergencePlot(p, J, epsilonArray, maxErrorArray) -> None:
    plt.scatter(epsilonArray, maxErrorArray, label='Maximum Errors')

    # plotting a trendline to observe slope
    coeffs = np.polyfit(np.log10(epsilonArray),np.log10(maxErrorArray),1)
    slope = coeffs[0]; intercept = coeffs[1]
    fitLine = 10**intercept * epsilonArray**slope
    plt.plot(epsilonArray, fitLine, linestyle='--', label=f'Slope: {slope:.3f}')

    plt.title(fr'Maximum Error vs. $\varepsilon$ for p={p} and J={J}')
    plt.xlabel(fr'$\varepsilon$'); plt.xscale('log')
    plt.ylabel(fr'$||\cdot||_\infty$'); plt.yscale('log')
    plt.legend()

    plt.show()