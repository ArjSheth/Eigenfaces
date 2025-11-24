import numpy as np
from SVD import SVD as svd
from PIL import Image as img
import glob
from MeanAdjustment import ColumnMeanCentering as cmc
from matplotlib import pyplot as plt
from Helper_Fns import ID


def concatrows(A : np.array) : # flattens an image with rows r1,r2,...,rn as r1+r2+...+rn. returns a LIST!
    L = []
    for rows in A :
        L.extend(rows)
    # print(f"Concatenated rows. Matrix was {A.shape}, list has lengeth {len(L)}")
    return L



image_files = glob.glob("/home/arjun/Pictures/Collect_Faces/*.jpg")  # Adjust extension as needed
images = []

for file in image_files:
    imgs = img.open(file)
    imgs = imgs.convert('L')
    mtx = np.array(imgs,dtype=float) # 
    images.append(concatrows(mtx))


images_array = np.array(images).T # This is LONG/SKINNY

mtx, mean_face = cmc(images_array)


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


def Unwrap(A : np.array) :
    B = np.reshape(A,newshape=(200,180)).copy()
    return B




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


plot_eigenfaces(U, (200, 180), k=20, grid=(4,4))


def UUt(x) :
    global U
    intermit = U.T@x
    return U@intermit


test_img1 = img.open("/home/arjun/Pictures/Collect_Faces/rmcoll.15.jpg")
test_img2 = img.open("/home/arjun/Pictures/Collect_Faces/ndhagu.13.jpg")
test_img3 = img.open("/home/arjun/Downloads/DrawnTestFace.png")
test_img1 = test_img1.convert('L')
test_img2 = test_img2.convert('L')
test_img3 = test_img3.convert('L')
tt_img1 = concatrows(np.array(test_img1, dtype=float))
tt_img2 = concatrows(np.array(test_img2, dtype=float))
tt_img3 = concatrows(np.array(test_img3, dtype=float))
tt_img = (7*np.array(tt_img1)+3*np.array(tt_img2))/10

back = U.T@tt_img #
back = [back[k]/S[k,k] for k in range(len(S))]

intermit = Unwrap(UUt(tt_img))
fin = np.clip(intermit, 0, 255)
fin = np.rint(fin).astype(np.uint8)
re_expressed = img.fromarray(fin)
re_expressed.save("Test_img_re_expressed.jpg")


np.save('PCA_FR/MeanFace.npy', mean_face)
np.save('PCA_FR/E_Faces.npy', U)
np.save('PCA_FR/SingVals.npy',S)
np.save('PCA_FR/V_mtx_from_USV.npy', V)

print("Saved mean face, and Eigenfaces in separate files.")
print("test saved")