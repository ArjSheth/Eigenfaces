import numpy as np
import glob
from PIL import Image as img
import matplotlib.pyplot as plt

"""
Load the eigenfaces from E_Faces.npy
Given an image, 'prepare' it.
Subtract mean face. Do limited UU* on it.
Save parameters to a CSV or something.
"""

def concatrows(A : np.array) : # flattens an image with rows r1,r2,...,rn as r1+r2+...+rn. returns a LIST!
    L = []
    for rows in A :
        L.extend(rows)
    # print(f"Concatenated rows. Matrix was {A.shape}, list has lengeth {len(L)}")
    return L

def Unwrap(A : np.array) :
    B = np.reshape(A,newshape=(200,180)).copy()
    return B



U = np.load('PCA_FR/E_Faces.npy') # U is skinny. Its columns are faces.
S = np.load('PCA_FR/SingVals.npy') # use for USV
MeanFace = np.load('PCA_FR/MeanFace.npy') # MeanFace is a column.

ajflem6 = img.open("/home/arjun/Pictures/faces94/male/ajflem/ajflem.6.jpg")
ajflem19 = img.open("/home/arjun/Pictures/faces94/male/ajflem/ajflem.19.jpg")
testface = img.open("/home/arjun/Pictures/Collect_Faces/rmcoll.15.jpg")
imgs = [ajflem6,ajflem19,testface]
e = []
for k in range(3) :
    imgs[k] = imgs[k].convert('L')
    imgs[k] = concatrows(np.array(imgs[k], dtype=float))


# Now testface is long. remove meanface.



newU = U[:, :113].copy()
for k in range(3) : 
    imgs[k] = (newU.T)@(imgs[k])
    e.append(imgs[k])
    imgs[k] = newU@imgs[k]

# We are good thus far.
# prepTest3 is UU*(testface-meanface)


# intermit = Unwrap(face)
# fin = np.clip(intermit, 0, 255)
# fin = np.rint(fin).astype(np.uint8)

# re_expressed = img.fromarray(fin)
# re_expressed.save("PCA_FR/Test_img_re_expressed.jpg")


def plot_eigenfaces(U, image_shape, k=16, grid=(4,4), cmap='gray'):
    fig, axes = plt.subplots(grid[0], grid[1], figsize=(2*grid[1], 2*grid[0]))
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


a = np.array(imgs).T
print(np.shape(a))

plot_eigenfaces(a, (200, 180), k=3, grid=(5,5))
print(np.linalg.norm(e[0]-e[1]))
print(np.linalg.norm(e[0]-e[2]))
