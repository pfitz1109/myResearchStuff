""" Little code to run some debugging stuff, easier to do it this way imo """

from utilities import wavelet_coefficient_generator, _compute_filter_coefficients
import numpy as np 

coarseX1 = np.linspace(0, 2*np.pi, 13)
refinedX1 = np.linspace(0, 2*np.pi, 25)
coarseF1 = np.sin(coarseX1); refinedF1=np.sin(refinedX1)

coarseX2 = refinedX1
refinedX2 = np.linspace(0, 2*np.pi, 49)
coarseF2 = refinedF1; refinedF2 = np.sin(refinedX2)

coef = _compute_filter_coefficients(6)

d1, plottingF1 = wavelet_coefficient_generator(6, 1, 2, 0.00001, coef, coarseF1, refinedF1)
d2, plottingF2 = wavelet_coefficient_generator(6, 2, 2, 0.00001, coef, coarseF2, refinedF2)

print(d1); print(d2)