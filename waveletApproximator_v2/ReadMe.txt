""" WAVELET APPROXIMATOR VERSION 2 """

This version of the code will take things a step further by generating the required forward wavelet transformation matrix "F" and the 
backward wavelet transformation matrix B. This should greatly increase computation time and allow for an easier computation of the
derivative matrices "gamma" that are necessary for true wavelet analysis. 

We will reuse some of the code from the previous version, but not all of it. New programs are needed to compute F and B, as well as the
connection coefficient matrices "Gamma" for a given funciton. This will require eigenvalue problems to be solved.