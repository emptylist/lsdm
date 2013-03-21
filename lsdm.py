#!/usr/bin/env python

import numpy as np
import numpy.linalg as la
from scipy import exp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

## TODO: CML file access to data

def gaussianKernel(x,y):
    return (-1*exp(abs(x-y))/2)

def constructDistanceMatrix(dataArray, eps):
    return squareform((1/eps) * pdist(dataArray, gaussianKernel)) 

def constructMarkovMatrix(distanceMatrix):
    pass

## TODO: eigenvalue/eigenvector computation


