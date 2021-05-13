import netpbmfile as nt
import numpy as np
from os import getcwd, chdir, mkdir
import scipy as sp
import matplotlib.pyplot as plt
import cv2
import skimage as sk
from skimage import filters
chdir("D:/Cours CS/Projet S8/MIAS Database/all-mias")
# image=nt.imread('mdb001.pgm')
# print(len(image))
# print(len(image[0]))
with nt.NetpbmFile('mdb001.pgm') as pgm:
    print(pgm.axes)
    print(pgm.shape)
    print(pgm.dtype)
    print(pgm.maxval)
    print(pgm.magicnum)
    image = pgm.asarray()
    print(image.shape)


def noisy(noise_typ,image):

    if noise_typ == "gauss":
        row,col= image.shape
        mean = 0
        var = 0
        sigma = 0.05
        gauss = np.random.normal(mean,sigma,(row,col))
        #gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.2
        out = np.array(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col = image.shape
        var=1
        sig=var**0.5
        gauss = np.random.normal(0,sig,(row,col))
        #gauss = gauss.reshape(row,col)        
        noisy = image + image * gauss
        return noisy


#plt.imshow(noise)
#plt.show()
#image=np.array(image)
#image[0][0]=1
#print(image[0][0])

def mean_filter(image):
    n,m=image.shape
    filtered=np.zeros((n,m))
    for i in range(1,n-1):
        for j in range(1,m-1):
            u=0
            for k in range(i-1,i+2):
                for h in range(j-1,j+2):
                    u+=image[k][h]
            filtered[i][j]=u/9
    return filtered

#filt=mean_filter(noise)


def median_filter(image):
    n,m=image.shape
    filtered=np.zeros((n,m))
    for i in range(1,n-1):
        for j in range(1,m-1):
            u=[]
            for k in range(i-1,i+2):
                for h in range(j-1,j+2):
                    u.append(image[k][h])
            u.sort()                        
            filtered[i][j]=u[4]
    return filtered

#filt=median_filter(image)
#plt.imshow(filt)

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
    filtered = np.multiply(filtered, 2.0)
    filtered = np.clip(filtered, 0, 255)
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

camera=sk.data.camera()
noise=noisy("s&p",image)


#adaptive_threshold = filters.threshold_local(image, 151)
#Metrics
def MSE(image,filt):
    n,m=image.shape
    s=0
    for i in range(n):
        for j in range(m):
            s+=(filt[i][j]-image[i][j])**2
    return s/(n*m)

def PSNR(image,filt):
    d=image.max()
    return 10*np.log10((d**2)/MSE(image,filt))

#print(MSE(image,filt1))
#print(MSE(image,filt2))
#print(MSE(image,filt3))


filt1=filtre_conv('binomial',noise)
filt2=filtre_conv('mean',noise)
filt3=median_filter(noise)
print(PSNR(image,filt1))
print(PSNR(image,filt2))
print(PSNR(image,filt3))
plt.figure(figsize=(16, 4))
plt.subplot(231)
plt.imshow(image,cmap = plt.get_cmap('gray'),vmin=image.min(),vmax=image.max())
plt.title("Image originale")
plt.subplot(232)
plt.imshow(filt1,cmap = plt.get_cmap('gray'),vmin=filt1.min(),vmax=filt1.max())
plt.title("Filtre binomial")
plt.subplot(233)
plt.imshow(filt2,cmap = plt.get_cmap('gray'),vmin=filt2.min(),vmax=filt2.max())
plt.title("Filtre moyen")
plt.subplot(234)
plt.imshow(filt3,cmap = plt.get_cmap('gray'),vmin=filt3.min(),vmax=filt3.max())
plt.title("Filtre médian")
plt.subplot(235)
plt.imshow(noise,cmap = plt.get_cmap('gray'),vmin=noise.min(),vmax=noise.max())
plt.title("Bruit salt and pepper")

#filt1=filtre_conv('xsobel',noise)
#filt2=filtre_conv('ysobel',noise)


#plt.figure(figsize=(16, 4))
#plt.subplot(311)
#plt.imshow(image,cmap = plt.get_cmap('gray'),vmin=image.min(),vmax=image.max())
#plt.title("Image originale")
#plt.tight_layout(pad=0.5)
#plt.subplot(312)
#plt.imshow(filt1,cmap = plt.get_cmap('gray'),vmin=filt1.min(),vmax=filt1.max())
#plt.title("Filtre de Sobel : formes horizontales")
#plt.tight_layout(pad=0.5)
#plt.subplot(313)
#plt.imshow(filt2,cmap = plt.get_cmap('gray'),vmin=filt2.min(),vmax=filt2.max())
#plt.title("Filtre de Sobel : formes verticales")



#print(filt.min(),filt.max())
#filt=median_filter(noise)
#plt.imshow(image,cmap = plt.get_cmap('gray'),vmin=image.min(),vmax=image.max())
#plt.imshow(noise,cmap = plt.get_cmap('gray'),vmin=noise.min(),vmax=noise.max())

#plt.imshow(filt)


#plt.imshow(adaptive_threshold,cmap = plt.get_cmap('gray'))
plt.show()

