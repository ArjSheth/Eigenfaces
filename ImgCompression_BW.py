from SVD import SVD as svd1
import glob
import PIL.Image as img
import numpy as np


def svd2(A) :
    u,s,v = np.linalg.svd(A)
    S = np.zeros(shape = (len(s), len(s)), dtype=float)
    for i in range(len(s)) :
        S[i,i] = s[i]
    return u,S,v
svd = svd2 # choose between svd1, svd2




check = 0
src = "/home/arjun/Pictures/Collect_Faces/9332898.3.jpg"

# src = input("Please enter path of chosen image : ")
def makeArr(src) :
    image = img.open(src)
    image = image.convert('L')
    return np.array(image, dtype=float)


scale = 2
init_arr = makeArr(src)[::scale, ::scale]

u,s,v = svd(init_arr)

print(f"reached checkpoint {check}, image SVD is done;\n")
check += 1

compression_factor = float(input("Provide the desired compression ratio (between 0 and 1) : \n"))
Sings = [s[i,i] for i in range(len(s))]
rank = len(Sings)

sum_of_sigmas = 0
for sval in Sings :
    sum_of_sigmas += sval
# Now I have a sum of singular values.

print(f"reached checkpoint {check}, evaluated total possible weight of singular values;\n")
check += 1


keep = 0
summ = 0
while keep < rank and summ/sum_of_sigmas < compression_factor :
    summ += Sings[keep]
    keep+=1

print(f"reached checkpoint {check}, determined how many singular values to keep;\n")
check += 1

U = u[:,:keep].copy()
S = s[:keep, :].copy()
V = v.copy()

print(f"reached checkpoint {check}, kept {keep} of {rank} singular values;\n")
check += 1
resultingArray = U@S@V


norm_diff = np.linalg.norm(init_arr - resultingArray)
print(f"reached checkpoint {check}, norm difference between the images as arrays (or frobenius norm of image vectors) is {norm_diff}.\n")


resultingImg = img.fromarray(resultingArray)
resultingImg.show()