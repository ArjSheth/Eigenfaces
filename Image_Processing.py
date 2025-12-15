import numpy as np
from PIL import Image as img
from MeanAdjustment import ColumnMeanCentering as cmc
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces as oli
from typing import Any


def load_imgs() :
    dataset : Any = oli() # This is done only because the type-checking is strict and i don't think it's needed here in particular.
    return dataset.data.T

def svd(A) :
    u,s,v = np.linalg.svd(A)
    S = np.zeros(shape = (len(s), len(s)), dtype=float)
    for i in range(len(s)) :
        S[i,i] = s[i]
    return u,S,v

def concatrows(A : np.ndarray) : # flattens an image with rows r1,r2,...,rn as r1+r2+...+rn. returns a LIST!
    L = []
    for rows in A.T :
        L.extend(rows)
    # print(f"Concatenated rows. Matrix was {A.shape}, list has lengeth {len(L)}")
    return L



# image_files = glob.glob("/home/arjun/Pictures/Collect_Faces/*.jpg")  # Adjust extension as needed
# images = []

# for file in image_files:
#     imgs = img.open(file)
#     imgs = imgs.convert('L')
#     mtx = np.array(imgs,dtype=float) # 
#     images.append(concatrows(mtx))
# imgs_list = []
# for imgs in glob.glob("/home/arjun/Pictures/Collect_Faces/*") :
#     imag = img.open(imgs)
#     imar = np.array(imag, dtype=float)
#     imgs_list.append(concatrows(imar))

# images_array = np.array(imgs_list).T # This is LONG/SKINNY
images_array = load_imgs()
use_imgs = images_array[:,:]
""" Yo what  """
mtx, mean_face = cmc(use_imgs)


# 1440 x 20 
# print(np.shape(mtx))
r, c = np.shape(mtx)
# subsample every 5th row/col to get a reduced image (no manual loops)


# print("subsampled shape:", hehu.shape)


U,S,V = svd(mtx)


# for i in range(k, n_diag):
#     S2[i][i] = 0.0



# new_mtx = U @ S2 @ V
# new_mtx = np.clip(new_mtx, 0, 255)
# new_mtx = np.rint(new_mtx).astype(np.uint8)

# new_img = img.fromarray(new_mtx)
# new_img.save("Long_Images_Mtx.png")
# print("saved")



prepU1 = np.clip(-U,0,255)
arr_img_norm = (prepU1 - prepU1.min())/(prepU1.max() - prepU1.min())
prepU2 = (arr_img_norm*255).astype(np.uint8)


def Unwrap(A : np.ndarray) :
    B = np.array(A,(64,64)).copy()
    return B




def plot_eigenfaces(U, image_shape, k=16, grid=(4,4), cmap='gray'):
    fig, axes = plt.subplots(grid[0], grid[1], figsize=(2*grid[1], 2*grid[0]))
    assert isinstance(axes, np.ndarray)
    for i, ax in enumerate(axes.flat):
        if i >= k:
            ax.axis('off')
            continue
        # Reshape, normalize to [0, 1] for visualization
        arr = U[:, i].reshape(image_shape)
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
        ax.imshow(arr_norm, cmap=cmap)
        ax.set_title(f'Eigenface {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plot_eigenfaces(U, (64,64), k=20, grid=(4,4))


np.save('MeanFace.npy', mean_face)
np.save('E_Faces.npy', U)
np.save('SingVals.npy',S)
np.save('V_mtx_from_USV.npy', V)

print("Saved mean face, and Eigenfaces in separate files.")

print("test saved")
