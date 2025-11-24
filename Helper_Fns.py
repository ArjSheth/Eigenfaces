import numpy as np
from SVD import SVD as svd
from PIL import Image as img
import glob
from MeanAdjustment import ColumnMeanCentering as cmc
import matplotlib.pyplot as plt


"""
Need code to make image "skinny".
Need code to make skinny images "proper" again.
Code ID


"""


Fixed_Shape = (200,180)


def makeSkinny(A : np.array) : # flattens an image with rows r1,r2,...,rn as r1+r2+...+rn. returns a LIST!
    L = []
    for rows in A :
        L.extend(rows)
    print(f"Concatenated rows. Matrix was {A.shape}, list has lengeth {len(L)}")
    return L

def makeProper(col : np.array) :
    if len(col) != Fixed_Shape[0]*Fixed_Shape[1] :
        raise IndexError
    return np.reshape(col, Fixed_Shape)


def ID(imgCol, U) :
    S = (U.T)@imgCol
    L = []
    for i in range(len(U)) :
        L.append(S[i,i])
    return L
