import numpy as np
import matplotlib.image as mpimg
from os import getcwd, chdir, mkdir
import scipy as sp
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
from matplotlib.pyplot import imread 
import cv2
import numpy as np
from pylab import ginput
from scipy import signal
import matplotlib
from scipy.ndimage import convolve 
#from ipywidgets import *

#matplotlib inline 
plt.rcParams["figure.figsize"] = (12, 10)
chdir("D:/Cours CS/Projet S8/MIAS Database/all-mias")
camera=cv2.imread('mdb001.pgm',0)
img = cv2.imread('mdb002.pgm',0)

###
###Spectrale
###
def filtpb_gauss(A, fc):
    M, N= A.shape
    # Image dans le domaine fréquentiel
    AA = np.fft.fftshift(np.fft.fft2(A)) 
    M0 = np.ceil((M+1) / 2)
    N0 = np.ceil((N+1)/2)
    U, V = np.mgrid[1:M+1, 1:N+1]
    D2 = (U - M0)**2 + (V - N0)**2
    # Réponse fréquentielle du filtre gaussien
    HH = np.exp(-D2 / (2 * fc**2))
    # Application du filtre et retour au domaine spatial
    BB = np.fft.ifftshift(AA*HH) 
    B = np.fft.ifft2(BB)
    B = np.real(B)
    return B

def filtpb_butter(A, fc, ordre): 
    M,N=A.shape
    # Image dans le domaine fréquentiel
    AA = np.fft.fftshift(np.fft.fft2(A)) 
    M0 = np.ceil((M+1) / 2)
    N0 = np.ceil((N+1) / 2)
    U, V = np.mgrid[1:M+1, 1:N+1]
    D2 = (U - M0)**2 + (V - N0)**2
# Réponse fréquentielle du filtre Butterworth
    HH = 1 / (1 + (D2 / fc**2)**ordre)
# Application du filtre et retour au domaine spatial
    BB = np.fft.ifftshift(AA * HH) 
    B = np.fft.ifft2(BB)
    B = np.real(B)
    return B
 

#%matplotlib inline plt.rcParams["figure.figsize"] = (12, 10)

dpiGlobal = 100
def sliders_gauss(image,a=1, fc=40):
    pb_gauss = filtpb_gauss(image, fc)
    img_gauss = image + a*(image-pb_gauss)
    #plt.figure(dpi=dpiGlobal)
    return img_gauss
#slider1 = interactive(sliders_gauss, a=(1, 3, 0.5), fc=(1, 50, 5)) 
#display(slider1)
def sliders_butter(image,a=3, fc=46, ordre=2):
    pb_butter = filtpb_butter(image, fc, ordre)
    img_butter = image + a*(image-pb_butter)
    #plt.figure(dpi=dpiGlobal)
    return img_butter
#slider2 = interactive(sliders_butter, a=(1, 3, 0.5), fc=(1, 50, 5),ordre=(1,10,1))
#figure(2)
#img_gauss = img + (img - filtpb_gauss(img, 26))
#img_butter = img + 3*(img - filtpb_butter(img, 46, 2))
img_gauss=img+10*(img-filtpb_gauss(img, 26))
img_butter=img+10*(img-filtpb_butter(img, 46, 2))
plt.subplot(131); plt.imshow(img_gauss, cmap="gray",vmin=0,vmax=255,interpolation="none")
plt.title('Filtre gaussien (a=1,fc=26)')
plt.subplot(132); plt.imshow(img, cmap="gray",vmin=0,vmax=255,interpolation="none")
plt.title('Image originale')
plt.subplot(133); plt.imshow(img_butter, cmap='gray', vmin=0, vmax=255,interpolation="none")
plt.title('Filtre Butterworth (a=3,fc=46)')
plt.show()

def unsharp_filter(img, taille_masque, coefficient_de_rehaussement): 
    a = coefficient_de_rehaussement
    Fprime = cv2.blur(img, (taille_masque, taille_masque)) 
    norm_Fprime = Fprime / np.max(Fprime)
    return img + a * (img - norm_Fprime)

def sliders_spatial(a=3, taille=5):
    img_unsharp = unsharp_filter(img/np.max(img),taille,a)*255 
    plt.imshow(img_unsharp, cmap="gray",vmin=0,vmax=255, interpolation="none") 
    plt.title(f'Unsharp filtering (a={a},taille={taille})')
    plt.show()
#slider3 = interactive(sliders_spatial, a=(0, 3, 1), taille=(3, 40, 2)) 
#figure(3)
img_unsharp = unsharp_filter(img/np.max(img), 5, 3)*255 
plt.subplot(121); 
plt.imshow(img, cmap="gray",vmin=0,vmax=255,interpolation="none")
plt.title('Image originale')
plt.subplot(122); plt.imshow(img_unsharp, cmap='gray', vmin=0, vmax=255,interpolation="none")
plt.title('Version rehaussée (a=3, taille=5)') 
plt.show()

####
####Laplacien
####

def laplacien(A):
    #normalisation:
    A = A/np.max(A) 
    M,N=A.shape
    # Image dans le domaine fréquentiel
    AA = np.fft.fftshift(np.fft.fft2(A)) 
    M0 = np.ceil((M+1) / 2)
    N0 = np.ceil((N+1) / 2)
    U, V = np.mgrid[1:M+1, 1:N+1]
    D2 = (U - M0)**2 + (V - N0)**2
# Réponse fréquentielle du laplacien
    HH = 4*(3.14159265359**2)*D2
    # Application du filtre et retour au domaine spatial
    BB = np.fft.ifftshift(AA * HH) 
    B = np.fft.ifft2(BB)
    B = np.real(B)
    return B
def sliders_laplacien(c=80):
    lap = laplacien(img)
    img_laplace = img + c*lap/np.max(lap)
    plt.imshow(img_laplace, cmap="gray",vmin=0,vmax=255, interpolation="none")
    plt.title(f'Rehaussement par laplacien (spectral,c={c})')
    plt.show()
#slider4 = interactive(sliders_laplacien, c=(0, 100, 5)) 
#display(slider4)
lap = laplacien(img)
img_laplace = img + 1000*lap/np.max(lap)
plt.subplot(121); 
plt.imshow(img, cmap="gray",vmin=0,vmax=255,interpolation="none")
plt.title('Image originale')
plt.subplot(122); 
plt.imshow(img_laplace, cmap='gray',vmin=0,vmax=255,interpolation="none")
plt.title('Image rehaussée (Laplacien, c=80)') 
plt.show()



noyau = np.ones((3,3)) 
noyau[1][1] = -8
# convolution avec le noyau
lap2 = cv2.filter2D(img/np.max(img), -1, noyau)
def sliders_laplacien(c=80):
    img_laplace2 = img - c*lap2/np.max(lap2)
    plt.figure(dpi=dpiGlobal)
    plt.imshow(img_laplace2, cmap="gray",vmin=0,vmax=255, interpolation="none") 
    plt.title(f'Rehaussement par laplacien (spatial, c={c})')
    plt.show()
#slider5 = interactive(sliders_laplacien, c=(0, 100, 5)) 
#display(slider5)
img_laplace2 = img - 80*lap2/np.max(lap2)
plt.subplot(121); 
plt.imshow(img, cmap="gray",vmin=0,vmax=255,interpolation="none")
plt.title('Image originale')
plt.subplot(122); 
plt.imshow(img_laplace2, cmap='gray',vmin=0,vmax=255,interpolation="none")
plt.title('Image rehaussée (Laplacien spatial, c=80)') 
plt.show()

lap = laplacien(img)
img_laplace = img + 80*lap/np.max(lap) 
img_laplace2= img - 80*lap2/np.max(lap2)
plt.figure(dpi=dpiGlobal)
plt.subplot(121);
plt.imshow(img_laplace, cmap="gray",vmin=0,vmax=255, interpolation="none") 
plt.title('Laplacien (spectral,c=80)')
plt.subplot(122);
plt.imshow(img_laplace2, cmap="gray",vmin=0,vmax=255, interpolation="none") 
plt.title('Laplacien (spatial,c=80)')
plt.show()





img_laplace = img + 80*lap/np.max(lap)
img_unsharp = unsharp_filter(img/np.max(img),5,2)*255
plt.figure(dpi=dpiGlobal)
plt.subplot(121);
plt.imshow(img_laplace, cmap="gray",vmin=0,vmax=255, interpolation="none") 
plt.title('Laplacien (spectral,c=80)')
plt.subplot(122);
plt.imshow(img_unsharp, cmap="gray",vmin=0,vmax=255, interpolation="none") 
plt.title('Unsharp (a=2,b=5)')
plt.show()