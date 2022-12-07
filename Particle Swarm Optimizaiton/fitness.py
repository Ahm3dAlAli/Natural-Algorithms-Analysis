#	Ahmed Al Ali
#	Analysis of Particle Swarm Optimization
#   19-Nov-2022
#   Natural Computing , University of Edinburgh
#########################################################################
import math
import numpy as np

def sphere(X,d):
    f=0
    for i in range(d-1):
        f = f+X[i]**2
    return f

def rastrigin(X,d):

    f=10*(d-1)
    for i in range(d-1):
        f = f+( X[i]**2 - 10*np.cos(2 * math.pi * X[i]))
    return f


