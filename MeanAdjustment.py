import numpy as np

def addLists(L1,L2) :
    for i in range(len(L1)) :
        L1[i] += L2[i]
    return L1

def ColumnMean(A : np.array) :
    At = (A.T).copy()
    myvec = [0 for j in range(len(At[0]))]
    for cols in At :
        myvec = addLists(myvec,cols)
    n = len(At)
    for item in myvec :
        item = item/n
    return myvec

def ColumnMeanCentering(A : np.array) : 
    m,n = np.shape(A)
    avgcol = list(ColumnMean(A)) # as a row
    B = np.array([avgcol for k in range(n)])
    res = A-B.T
    return res, (B[0]).T