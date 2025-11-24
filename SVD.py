import QR_factorization as qrfac
import numpy as np
import QR_Algo as alt




def sortsigmas(diag : list) :
    length = len(diag)
    sortedl = sorted(diag)[::-1]
    perm_list = [sortedl.index(diag[i]) for i in range(length)]
    P = np.zeros(shape = (length,length))
    for k in range(length) :
        P[k][perm_list[k]] = 1.0
    return P
    


def SVD(A):
    r,c = np.shape(A)
    if r<c :
        U,S,V = SVD(A.T)
        return V.T, S, U.T
    else :
        pass
    AtA = A.T@A
    UH, P = qrfac.UHess(AtA)
    #P.UH.Pt = AtA
    Q,R = alt.qr_Algo(UH)
    ## Q.R.Q* = UH = P.AtA.Pt = P.U.S2.V.P
    # QR = P.AtA.PQ
    V = Q.T@P.T
    S = np.identity(c, dtype=qrfac.dd)
    for i in range(c) :
        rii = R[i][i]
        if np.abs(rii) <= 1.0e-8 :
            S[i][i] = 0.0
        else :
            S[i][i] = np.sqrt(np.abs(rii))
    U = np.zeros(shape=(c,r), dtype=qrfac.dd)
    AVt = A@V.T
    # print(f"A has size {np.shape(A)}, while U has {np.shape(U)}")
    L = []
    for i in range(c) :
        if S[i][i] == 0.0 :
            U[i] = [0.0 for k in range(r)]
        else :
            U[i] = [AVt[k][i]/S[i][i] for k in range(r)]
        L.append(S[i][i])
    U = U.T
    
    # The following gives us S with sorted singular values. `Permu` is the matrix corresponding to the permutation.
    Permu = sortsigmas(L)
    U = U@Permu.T
    S = Permu@S@Permu.T
    V = Permu@V
    return U,S,V
