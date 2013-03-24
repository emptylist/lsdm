#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
from scipy import exp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

## TODO: CML file access to data

'''Currently this is acting as a function library.

Furthermore this only implements Diffusion Maps,
*NOT* Locally Scaled DMs at this time.'''

def constructProbabilityKernel(dataArray, eps):
    '''Generates a distance/probability matrix from the gaussian
    kernel.  The probability matrix is not normalized, and does
    not represent actual probabilities.  This is the major bottleneck
    in the performance thus far.  Using kd-trees from either the 
    scikit-ann or scikit-learn packages should help this.'''
    return squareform((1/(2*eps)) * pdist(dataArray, 'sqeuclidean')) 

def normalizeProbabilityKernel(probabilityKernel):
    '''Normalizing the prob matrix is equivalent to 
    approximating a density estimator under the 
    kernel used for the distance function.  Small experiments
    suggest this is efficient.'''
    P = np.apply_along_axis(np.sum, 1, probabilityKernel)
    return probabilityKernel / np.sqrt(np.outer(P,P))

def constructMarkovMatrix(npKernel):
    '''This produces a row-stochastic matrix from a positive
    semi-definite matrix.'''
    return np.apply_along_axis(lambda v: v/sum(v), 1, npKernel)

def diffusionMap(markovMatrix):
    '''Performs eigenvalue decomposition on the Markov matrix.
    This needs to be replaced with sparse methods and return
    and object with the embedded data instead.'''
    ## TODO: Return embedding instead.
    return np.eig(markovMatrix)

if __name__ == "__main__":
    print "TODO: Add commandline functionality!"
