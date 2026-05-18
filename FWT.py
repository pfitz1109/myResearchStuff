import numpy as np

def FWT(p,j):
    hSquiggle = np.zeros((2**j*p+1,2**(j+1)*p+1))
    gSquiggle = np.zeros((2**j*p,2**(j+1)*p+1))
    

    return np.append(hSquiggle,gSquiggle,axis=0)