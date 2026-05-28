""" PLOTTING UTILITIES """
""" Specific plotting functions for the wavelet approximation toolkit """

import numpy as np
import matplotlib.pyplot as plt

# tool for plotting the sparse wavelet approximation against the analytical solution 

# entires in each are structured as follows:
# plottingX = [ [s0_locations], [significant_d1_locations], [significant_d2_locations], ...]
# plottingF = [ [s0_values], [function_values_at_significant_d1_locations, [function_approx_at_signficant_d2_locations], ... ]
# here "significant" means that the wavelet coefficient at the x-coordinate was greater than the threshold
# essentially, this is a sparse wavelet approximation
def waveletApproximationPlotting(func, plottingX, plottingF, eps, left_bound, right_bound) -> None:

    # generates scatter plot of dense wavelet approximations 
    plt.scatter(plottingX, plottingF, label ='Wavelet Approximation')

    # generating and plotting analytical function
    functionalX = np.linspace(left_bound,right_bound,10000)
    functionalY = func(functionalX)
    plt.plot(functionalX,functionalY, label='Analytical Solution', color='r')

    # plot details
    plt.title(fr'Analytical vs Approximate for $\varepsilon$ ={eps}'); plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend();
    plt.show()

# plot the error across the domain for the highest resolutioon level achieved 
def waveletApproximationErrorPlotting(errorX, absoluteError, j_final, eps) -> None:
    # plotting absolute error across domain 
    plt.plot(errorX, absoluteError, label ='Absolute Error')
    # plot details 
    plt.xlabel('x'); plt.ylabel(r'$||\cdot||_\infty$'); plt.title(fr'Absolute Error at j={j_final} Resolution for $\varepsilon$={eps}')
    plt.show()

# NOT CURRENTLY WORKING 
def waveletResolutionLevelsPlotting(X0, xLevelsPlotting, levelsPlotting, eps) -> None:
    # scatter plot of the lowest-resolution grid points - putting this here so it appears first in the plot 
    plt.scatter(X0, np.zeros(len(X0)), label='Resolution Level j = 0')

    # have to convert to numpy arrays first
    xLevelsPlotting = np.concatenate(xLevelsPlotting); levelsPlotting=np.concatenate(levelsPlotting)

    plt.scatter(xLevelsPlotting, levelsPlotting, color='black')
    plt.xlabel('x'); plt.ylabel('Resolution Level'); plt.title(fr'Resolution Levels Across Grid for $\varepsilon$ = {eps}'); plt.legend(loc = 'best')
    plt.show()
