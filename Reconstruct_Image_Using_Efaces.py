import PIL.Image as img
import numpy as np

path = "/home/asheth/Desktop/TestImgsForCompression/img1.jpeg"
myprof = img.open(path)
myprof = myprof.convert("L")
myprof_mtx = np.array(myprof)
print(np.shape(myprof_mtx))
reshaped = myprof_mtx[:1280:20,:1280:20]
print(np.shape(myprof_mtx))


myprof_mtx = reshaped.reshape(4096)

U = np.load('FR/E_Faces.npy')
Mean = np.load('FR/MeanFace.npy')

Utprof = U.T@(myprof_mtx-Mean)
size = len(Utprof)

use_sings = 4000
rounded_Utprof = [round(Utprof[i], 1) for i in range(use_sings)] + [0 for i in range(len(Mean)-use_sings)]


recon = U@rounded_Utprof + Mean
recon = recon.reshape(64,64)
rounded_man = img.fromarray(recon)
rounded_man.show()