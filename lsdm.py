#!/usr/bin/env python

import numpy as np
import numpy.linalg as la

from concurrent import futures
import multiprocessing

import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy import exp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

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

def _sparseDistances(A,v,eps):
    # Testing shows apply_along_axis broadcasting is efficient
    # for the LLVM compiled rmsd distance function
    D = np.apply_along_axis(_pairwiseRMSD, 1, A, v)  ## Returns a vector
    threshold = exp(-1/(2*eps))
    D = _thresholdVector(D, eps)
    return scipy.sparse.coo_matrix(D)

####
#    Local Scaling Functions
####
"""Local scaling is hard, it turns out. This section is largely incomplete.
Scikit-learn's MDS routines seem inefficient.  Some Numba LLVM magic might be
in order.
"""
def _sortDistance(A,v):
    return sorted(zip(A, _pairwiseDistance(A,v), key=lambda i: i[1]))

def _localScaleDetermination(dataArray):
    pass

####
#    Diffusion Maps
####

def _constructDistanceMatrix(dataArray,eps):
    ## This is where we should do futures!
    sparse_vectors = []
    cpus = multiprocessing.cpu_count()

    def distances(v):
        return _sparseDistances(dataArray, v, eps)

    #The futures don't work; it seems that either Numba compiled or NumPy
    # functions don't play nice with the pickling process that is used
    # to do the multiprocessing.
    with futures.ThreadPoolExecutor(max_workers=cpus) as executor:
        for svec in executor.map(distances, dataArray):
            sparse_vectors.append(svec)
    #sparse_vectors = [_sparseDistances(dataArray, v, eps) for v in dataArray]
                                     
    return scipy.sparse.vstack(sparse_vectors).tocsr()

def _constructProbabilityKernel(dataArray, eps):
    '''Constructs the Markov matrix from a data array and scale parameter.'''
    # TODO: This function should be broken up so local scaling can be performed as well.
    K = _constructDistanceMatrix(dataArray, eps).todense()
    #P = scipy.sparse.csr_matrix(K.sum(1))
    P = K.sum(1)
    #print P.shape
    P = P*P.T
    #print P.shape
    P = 1./np.sqrt(P)
    #print P.shape
    Kbar = K * P
    #print Kbar.shape
    #print np.sqrt(1./Kbar.sum(1)).shape
    D = np.diag(np.sqrt(1./Kbar.sum(1)).T)
    #D = np.diag(np.sqrt(1./np.apply_along_axis(np.sum, 1, Kbar)))
    #print D.shape
    #print Kbar.shape
    return np.dot(np.dot(D,Kbar),D)

class DiffusionMap():
    '''Diffusion Map object styled on the scikit-learn SpectralEmbedding
    object. '''
    
    def __init__(self, n_components=2, local_scaling=False, eigen_solver=None
                 , epsilon=0.1):
        self.n_components = n_components
        self.local_scaling = local_scaling
        self.eigen_solver = eigen_solver
        self.epsilon = epsilon

    def _getAffinityMatrix(self, X):
        affinity_matrix = _constructProbabilityKernel(X, self.epsilon)
        if self.local_scaling:
            print "Local scaling is not implemented yet. Reverting to standard diffusion maps."
        return affinity_matrix

    def fit(self, X):
        """Fit the model from data in X."""
        self.affinity_matrix_ = self._getAffinityMatrix(X)
        lambdas, embedding = eigsh(self.affinity_matrix_,
                                   k = self.n_components + 1)
        self.embedding_ = embedding[:self.n_components]
        return self
    
    def fit_transform(self, X):
        """Fit the model from data in X and transform X."""

        self.fit(X)
        return self.embedding_
        
if __name__ == "__main__":
    print "TODO: Add commandline functionality!"
