#!/usr/bin/env python

import numpy as np
import numpy.linalg as la

import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy import exp

from numba.decorators import autojit

from functools import partial

#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.manifold.spectral_embedding import spectral_embedding, SpectralEmbedding

## TODO: CML file access to data

## ISSUE: It's currently not clear what the best interface is for integrating
## locally-scaled and fixed neighborhood diffusion maps into a single class

'''Currently this is acting as a function library.

Furthermore this only implements Diffusion Maps,
*NOT* Locally Scaled DMs at this time.

    References
    _________

    - Determination of reaction coordinates via locally scaled diffusion map, 2011
      Mary A. Rohrdanz, et al.
      http://dx.doi.org/10.1063/1.3569857

    - Diffusion Maps, Spectral Clustering and Eigenfunctions of Fokker-Planck Operators, 2005
      Boaz Nadler, et al.
      **Add ref location**

'''

####
#    Low-level and Distance Functions
####

'''Some low-level building blocks.  Most of it can be done more
generally using NumPy built-in functions, however, testing shows
that Numba LLVM compilation speeds up these small operations around
3-10x the NumPy versions.  As the distance function will be called
hundreds of thousands of times, this is quite an improvement.'''

@autojit
def _pairwiseRMSD(v,w):
    n = len(v)
    dist_sq = 0.0
    for i in xrange(n):
        dist_sq += (v[i] - w[i]) * (v[i] - w[i])
    return np.sqrt(dist_sq / n)

@autojit
def _thresholdVector(v, eps):
    for i in xrange(v.shape[0]):
        if v[i] < eps:
            v[i] = 0
    return v

####
#    Local Scaling Functions
####
"""Local scaling is hard, it turns out. This section is largely incomplete.
Scikit-learn's MDS routines seem inefficient (and we only really need SVDs anyway?)
Some Numba LLVM magic might be in order.
"""
def _sortDistance(A,v):
    return sorted(zip(A, _pairwiseDistance(A,v), key=lambda i: i[1]))

def _localScaleDetermination(dataArray):
    pass

####
#    Diffusion Maps
####

def _constructDistanceMatrix(dataArray):
    return np.apply_along_axis(lambda v: np.apply_along_axis(_pairwiseRMSD, 1, dataArray, v)
                               , 1, dataArray)

def _constructProbabilityKernel(dM):
    '''Constructs Kbar from a scaled distance matrix array..'''
    K = exp(-1*dM)
    P = K.sum(1)
    P = np.outer(P,P)
    P = 1./np.sqrt(P)
    Kbar = K * P
    return Kbar
#    D = np.diag(np.sqrt(1./Kbar.sum(1)).T)
#    return np.dot(np.dot(D,Kbar),D)

class DiffusionMap():
    '''Diffusion Map object styled on the scikit-learn SpectralEmbedding
    object. '''
    
    def __init__(self, n_components=2, local_scaling=False, eigen_solver=None
                 , epsilon=0.1):
        self.n_components = n_components
        self.local_scaling = local_scaling
        self.eigen_solver = eigen_solver
        self.epsilon = epsilon
        
    def _localScaling(self, dM):
        self.local_scale, self.local_dimension = _localScaling(dM)
        return self

    def _affinityMatrix(self, dM):
        if self.local_scaling:
            scale_factors = 1./self.local_scale
            dM = np.outer(scale_factors, scale_factors) * dM
        else:
            dM = (1./self.epsilon) * dM
        Kbar = _constructProbabilityKernel(dM)
        D = np.diag(np.sqrt(1./Kbar.sum(1)).T)
        self.affinity_matrix = np.dot(np.dot(D,Kbar),D)
        return self

    def fit(self, X):
        """Fit the model from data in X."""
        dM = _constructDistanceMatrix(X)
        self._affinityMatrix(dM)
        lambdas, embedding = eigsh(self.affinity_matrix,
                                   k = self.n_components + 1)
        self.embedding_ = embedding[:self.n_components]
        return self
    
    def fit_transform(self, X):
        """Fit the model from data in X and transform X."""

        self.fit(X)
        return self.embedding_
        
if __name__ == "__main__":
    print "TODO: Add commandline functionality!"
