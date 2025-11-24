import numpy as np
from SVD import SVD as svd


def PCA_tall (X : np.array, n=None) :
    # Assumes given matrix is tall/skinny
    u,s,v = svd(X)
    s1,s2 = np.shape(s)
    if n is not None and s1 < n :
        raise IndexError (f"Can't keep {s1} of {n} singular values")
    if n is None :
        return u,s,v
    uu = (u[:,:n]).copy()
    vv = (v[:n,:]).copy()
    ss = (s[:n,:n]).copy()
    return uu,ss,vv

def PCA_broad (X : np.array, n= None) :
    u,s,v = svd(X) # u is square. v is broad.
    s1,s2 = np.shape(s)
    if n is not None and s1 < n :
        raise IndexError (f"Can't keep {s1} of {n} singular values")
    if n is None :
        return u,s,v
    uu = (u[:,:n]).copy()
    vv = (v[:n,:]).copy()
    ss = (s[:n,:n]).copy()
    return uu,ss,vv


def PCA(X : np.array, idx = None) :
    m,n = np.shape(X)
    print(m,n)
    if m>=n :
        return PCA_tall(X,idx)
    return PCA_broad(X, idx)