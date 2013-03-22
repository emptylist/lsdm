#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
from scipy import exp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

## TODO: CML file access to data

## Currently this is functioning as a function library

## The slowest part of this flow is constructing the
## Probability Kernel.  Perhaps preprocessing the data
## to reduce the calls will help.  Or writing the
## metric function in C.

def gaussianKernel(x,y):
    v = x - y
    return exp(-1*(np.dot(v,v))/2)

def constructProbabilityKernel(dataArray, eps):
    return squareform((1/eps) * pdist(dataArray, gaussianKernel)) 

def normalizeProbabilityKernel(probabilityKernel):
    P = np.apply_along_axis(np.sum, 1, probabilityKernel)
    return probabilityKernel / np.sqrt(np.outer(P,P))

def constructMarkovMatrix(npKernel):
    return np.apply_along_axis(lambda v: v/sum(v), 1, npKernel)

def diffusionMap(markovMatrix):
    return np.eig(markovMatrix)

## TODO: Output the data in the embedded manifold
