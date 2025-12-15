import numpy as np


def svd(A) :
    u,s,v = np.linalg.svd(A)
    S = np.zeros(shape = (len(s), len(s)), dtype=float)
    for i in range(len(s)) :
        S[i,i] = s[i]
    return u,S,v

def PCA_tall (X : np.ndarray, n=None) :
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

def PCA_broad (X : np.ndarray, n= None) :
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


def PCA(X : np.ndarray, idx = None) :
    m,n = np.shape(X)
    print(m,n)
    if m>=n :
        return PCA_tall(X,idx)
    return PCA_broad(X, idx)