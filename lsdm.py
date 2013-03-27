#!/usr/bin/env python

import numpy as np
import numpy.linalg as la

from scipy import exp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.base import BaseEstimator, TransformerMixins
from sklearn.manifold.spectral_embedding import spectral_embedding, SpectralEmbedding

## TODO: CML file access to data

## Note to self: The Markov Matrix *IS* the normalized graph Laplacian
## under the diffusion distance kernel.  Hence it suffices to build the
## diffusion affinity matrix and simply pass that to scikit-learn's
## Spectral Embedding.

## Design question: Do I build a DiffusionMap object modeled on the
## SpectralEmbedding object in scikit-learn, or write functions that
## build the diffusion affinity matrix and pass that as a precomputed
## affinity matrix to Spectral Embedding?  The first has a nicer and
## more consistent interface, but the second prevents some unnecessary
## code replication (with essentially copy&paste style).

'''Currently this is acting as a function library.

Furthermore this only implements Diffusion Maps,
*NOT* Locally Scaled DMs at this time.'''

def _local_scale_determination(dataArray):
    pass

def _constructProbabilityKernel(dataArray, eps):
    '''Generates a distance/probability matrix from the gaussian
    kernel.  The probability matrix is not normalized, and does
    not represent actual probabilities.  This is the major bottleneck
    in the performance thus far.  Using kd-trees from either the 
    scikit-ann or scikit-learn packages should help this.'''
    return squareform(exp(-1*((1/(2*eps)) * pdist(dataArray, 'sqeuclidean'))))

def fitDiffusionMap(dataArray, eps, n_components=2):
    X = _constructProbabilityKernel(dataArray, eps)
    return SpectralEmbedding(n_components, affinity="precomputed").fit(X)

def fit_transformDiffusionMap(dataArray, eps, n_components=2):
    X = _constructProbabilityKernel(dataArray, eps)
    return SpectralEmbedding(n_components, affinity="precomputed").fit(X)

class DiffusionMap(BaseEstimator, TransformerMixins):
    '''Diffusion Map object styled on (and using code from) the scikit-learn
    SpectralEmbedding object. Unfinished.'''
    
    def __init__(self, n_components=2, affinity="precomputed",
                 gamma=None, random_state=None, eigen_solver=None,
                 n_neighbors=None):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.n_neighbors = n_neighbors
        


if __name__ == "__main__":
    print "TODO: Add commandline functionality!"
