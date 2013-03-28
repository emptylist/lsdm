#!/usr/bin/env python

import numpy as np
import numpy.linalg as la

from scipy import exp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold.spectral_embedding import spectral_embedding, SpectralEmbedding

## TODO: CML file access to data

## Note to self: The Markov Matrix *IS* the normalized graph Laplacian
## under the diffusion distance kernel.  Hence it suffices to build the
## diffusion affinity matrix and simply pass that to scikit-learn's
## Spectral Embedding class or spectral_embedding function.

## Design question: Do I build a DiffusionMap object modeled on the
## SpectralEmbedding object in scikit-learn, or write functions that
## build the diffusion affinity matrix and pass that as a precomputed
## affinity matrix to Spectral Embedding?  The first has a nicer and
## more consistent interface, but the second prevents some unnecessary
## code replication (with essentially copy&paste style).

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

def _local_scale_determination(dataArray):
    pass

def _constructProbabilityKernel(dataArray, eps):
    '''Generates a distance/probability matrix from the gaussian
    kernel.  The probability matrix is not normalized, and does
    not represent actual probabilities.  This is the major bottleneck
    in the performance thus far.  Using kd-trees from either the 
    scikit-ann or scikit-learn packages should help this.'''
    return squareform(exp(-1*((1/(2*eps)) * pdist(dataArray, 'sqeuclidean'))))

class DiffusionMap(BaseEstimator, TransformerMixin):
    '''Diffusion Map object styled on (and using code from) the scikit-learn
    SpectralEmbedding object. '''
    
    def __init__(self, n_components=2, local_scaling=False, eigen_solver=None):
        self.n_components = n_components
        self.local_scaling = local_scaling
        self.eigen_solver = eigen_solver

    def _get_affinity_matrix(self, X):
        affinity_matrix = _constructProbabilityKernel(X)
        if self.local_scaling:
            print "Local scaling is not implemented yet. Reverting to standard diffusion maps."
        return affinity_matrix


    def fit(self, X):
        """Fit the model from data in X."""
        self.affinity_matrix_ = self._get_affinity_matrix(X)
        self.embedding_ = spectral_embedding(affinity_matrix,
                                             n_components=self.n_components,
                                             eigen_solver=self.eigen_solver)
        return self
    
    def fit_transform(self, X):
        """Fit the model from data in X and transform X."""

        self.fit(X)
        return self.embedding_
        


if __name__ == "__main__":
    print "TODO: Add commandline functionality!"
