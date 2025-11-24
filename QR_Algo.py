import QR_factorization as qr
import numpy as np

def qr_Algo(M : np.array) :
    iter = 0
    Q , R = qr.QR(M) #QR is M
    KeepQ = Q
    while iter < 20 :
        new_mtx = R@Q
        Q,R = qr.QR(new_mtx)
        KeepQ = KeepQ@Q
        iter += 1
    return KeepQ, R # KeepQ.RKeepQ* = M