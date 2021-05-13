import netpbmfile as nt
import numpy as np
from os import getcwd, chdir, mkdir
import scipy as sp
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from skimage import filters
chdir("D:/Cours CS/Projet S8/MIAS Database/all-mias")

camera=sk.data.camera()

def Convolution2D(mask,image):
    n,m=image.shape
    filtered=np.zeros((n,m))
    c=int((len(mask)-1)/2)
    for i in range(c,n-c):
        for j in range(c,n-c):
            p=0
            for k in range(-c,c+1):
                for h in range(-c,c+1):
                    p += mask[k+1][h+1]*image[k+i,h+j]
    # normalisation des composantes
            filtered[i][j] = int(p)
  # retourne le pixel convolué
    return filtered

def filtre_conv(type,image):
    if type=='gauss':
        mask=[[1,2,1],[2,4,2],[1,2,1]]
    elif type=='mean':
        mask=[[1,1,1],[1,1,1],[1,1,1]]
    elif type=='id':
        mask=[[0,0,0],[0,1,0],[0,0,0]]
    elif type=='binomial':
        mask=[[1,4,6,4,1],[4,16,24,16,4],[6,30,48,30,6],[4,16,24,16,4],[1,4,6,4,1]]
    elif type=='ysobel':
        mask=[[-1,0,1],[-2,0,2],[-1,0,1]]
    elif type=='xsobel':
        mask=[[-1,-2,-1],[0,0,0],[1,2,1]]
    elif type=='4-laplacien':
        mask=[[0,1,0],[1,-4,1],[0,1,0]]
    elif type=='passe-haut':
        mask=[[0,-4,0],[-4,24,-4],[0,-4,-0]]
    elif type=='passe-bas':
        mask=[[2,2,2],[2,4,2],[2,2,2]]
    s=0
    for b in mask:
        for a in b:
            s+=a
    if s!=0:       
        mask=[[(1/s)*a for a in b] for b in mask]
    print(mask)
    return Convolution2D(mask,image)

plt.imshow(filtre_conv('4-laplacien',camera),cmap='gray')
plt.show()