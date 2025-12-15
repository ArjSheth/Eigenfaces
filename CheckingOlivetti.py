from sklearn.datasets import fetch_olivetti_faces as oli
import PIL.Image as img
import typing

def load_dat() :
    J : typing.Any = oli()
    return J.data # This is a.s. safe, because the database is verified. Trust, bbg.

L = []
for k in range(50) :
    imag = load_dat()[k]
    imag = imag.reshape((64,64))
    imag_uint8 = (imag * 255).astype("uint8")
    L.append(img.fromarray(imag_uint8, mode="L"))  # grayscale

import matplotlib.pyplot as plt
rows = 10
cols = 5
fig, axes = plt.subplots(rows, cols, figsize=(64,64))
axes = axes.ravel()  # Flatten the 2D array of axes for easy indexing

titles = [f'Image {i+1}' for i in range(50)]

for i in range (50):
    axes[i].imshow(L[i])
    axes[i].set_title(titles[i], fontsize=8)
    axes[i].axis('off')  # Hide axes ticks and labels

# Hide any unused subplots
for i in range(50, rows * cols):
    axes[i].axis('off')
plt.show()