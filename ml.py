import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras import optimizers
from keras import losses
from os import getcwd, chdir, mkdir
import os 
from tensorflow import Tensorboard
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import pandas as pd
chdir("D:/Cours CS/Projet S8/MIAS Database")
###

dataframe1 = pd.read_csv("Info.txt")
dataframe1.to_csv("Info.csv", index = None)

###

data_spec=pd.read_csv("Info.txt",sep=" ")
data_spec=data_spec.drop('Unnamed: 7',axis=1)
data_spec['SEVERITY']=data_spec['SEVERITY'].map({'B':'B','M':'M',np.nan:'S'},na_action=None)
print(data_spec)
data_spec.dropna(subset = ["SEVERITY"], inplace=True) 
data_spec.reset_index(inplace = True)
data_spec=data_spec.drop([3], axis=0)
data_spec.reset_index(inplace = True)
dspec = data_spec.REFNUM

###
chdir("D:/Cours CS/Projet S8/MIAS Database/all-mias")
lb = []
for i in range(len(data_spec)):
    if data_spec.SEVERITY[i] == 'B':
        lb.append(1)
    else:
        lb.append(0)
lb=np.array(lb)
img_name = []
for i in range(len(lb)):
        img_name.append(dspec[i]+ '.pgm')
img_name = np.array(img_name)

###

imgs=[cv2.imread(img_name[i],0) for i in range(len(lb))]
x_train, x_test, y_train, y_test = train_test_split(imgs, lb, test_size = 0.2, random_state = 42)
print(len(x_train))


print(imgs[72])
img=cv2.imread('mdb191.pgm')
plt.imshow(img)
plt.show()
print(len(lb))
### view_25_random_image():
plt.figure(figsize = (15, 10))
for i in range(25):
    rand = random.randint(0,len(lb)-1)
    plt.subplot(5, 5, i+1)
    
    img = imgs[rand]
    #img = cv2.resize(img, (256,256))
    if lb[rand] == 1:
        plt.title('B')
    else:
        plt.title('M')
    plt.tight_layout()
    plt.axis('off')
    try:
        plt.imshow(img)
    except Exception as e:
        print(rand)
        print(img_name[rand])    
#fig.savefig('random_25_image_fig.png')
plt.show()

###
rows,cols=224,224
img_train = []
lb_train = []
M=[]

ang=20
for angle in range(ang):
    M.append(cv2.getRotationMatrix2D(((cols-1) / 2, (rows-1) / 2), 4*angle, 1))
for i in range(len(x_train)):
    try : 
        img = cv2.resize(x_train[i],(rows,cols))
    except Exception as e:
        print(str(e))
    l=y_train[i]
    for angle in range(ang):
            img_rotated = cv2.warpAffine(img, M[angle], (224, 224))
            img_train.append(img_rotated)
            lb_train.append(l)
y_train = np.array(lb_train)
x_train = np.array(img_train)

print(x_train.shape)



###view rotated images
fig = plt.figure(figsize = (15, 10))
for i in range(25):
    rand = random.randint(0,len(y_train)-1)
    ax = plt.subplot(5, 5, i+1)
    
    img = x_train[rand]
    if y_train[rand] == 1:
        plt.title('B')
    else:
        plt.title('M')
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(img)
plt.show()

###
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 180,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.6, # Randomly zoom image 
        width_shift_range=0.5,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.5,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(x_train)

###
x_train=[cv2.resize(a,(rows,cols)) for a in x_train]
y_train = np.array(y_train)
x_train = np.array(x_train)
x_test=[cv2.resize(a,(rows,cols)) for a in x_test]
x_test = np.array(x_test)
y_test=np.array(y_test)
(a,b,c)=x_train.shape
x_train=np.reshape(x_train,(a,b,c,1))
(a,b,c)=x_test.shape
x_test=np.reshape(x_test,(a,b,c,1))
print(x_train.shape)

### model : 
def model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 1)))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

      
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


model = model()
model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='loss', mode='auto', patience=1,restore_best_weights=True, verbose=1)

check_point_filepath = 'D:/Cours CS/Projet S8/Project-S8-Mammography'

model_check_point = ModelCheckpoint(filepath =check_point_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')



hist = model.fit(x_train,
                 y_train,
                 epochs=5,
                 batch_size=64,
                 callbacks=[early_stop, model_check_point],validation_data=(x_test,y_test),initial_epoch=0)




model.evaluate(x_test,y_test)

###


###



chdir("D:/Cours CS/Projet S8/MIAS Database")



data_spec=pd.read_csv("Info.txt",sep=" ")
data_spec=data_spec.drop('Unnamed: 7',axis=1)
data_spec['SEVERITY']=data_spec['SEVERITY'].map({'B':'B','M':'M',np.nan:'S'},na_action=None)
print(data_spec)
dspec = data_spec.REFNUM

###
chdir("D:/Cours CS/Projet S8/MIAS Database/all-mias")
lb2 = []
for i in range(len(data_spec)):
    d=data_spec.SEVERITY[i]
    if  d== 'M':
        lb2.append(2)
    elif d=='B': 
        lb2.append(1)
    else:
        lb2.append(0)
lb2=np.array(lb2)
img_name2 = []
for i in range(len(lb2)):
        img_name2.append(dspec[i]+ '.pgm')
img_name2 = np.array(img_name2)

###

imgs2=[cv2.imread(img_name2[i],0) for i in range(len(lb2))]
x_train2, x_test2, y_train2, y_test2 = train_test_split(imgs2, lb2, test_size = 0.2, random_state = 42)
print(len(x_train2))


print(imgs2[72])
img=cv2.imread('mdb191.pgm')
plt.imshow(img)
plt.show()
print(len(lb2))
### view_25_random_image():
plt.figure(figsize = (15, 10))
for i in range(25):
    rand = random.randint(0,len(lb2)-1)
    plt.subplot(5, 5, i+1)
    
    img = imgs2[rand]
    #img = cv2.resize(img, (256,256))
    if lb2[rand] == 1:
        plt.title('B')
    elif lb2[rand]==2:
        plt.title('M')
    else:
        plt.title('S')
    plt.tight_layout()
    plt.axis('off')
    try:
        plt.imshow(img)
    except Exception as e:
        print(rand)
        print(img_name2[rand])    
#fig.savefig('random_25_image_fig.png')
plt.show()

###
rows2,cols2=224,224
img_train2 = []
lb_train2 = []
M2=[]

ang2=20
for angle in range(ang2):
    M2.append(cv2.getRotationMatrix2D(((cols-1) / 2, (rows-1) / 2), 4*angle, 1))
for i in range(len(x_train2)):
    try : 
        img = cv2.resize(x_train2[i],(rows2,cols2))
    except Exception as e:
        print(str(e))
    l=y_train2[i]
    for angle in range(ang2):
            img_rotated = cv2.warpAffine(img, M2[angle], (224, 224))
            img_train2.append(img_rotated)
            lb_train2.append(l)
y_train2 = np.array(lb_train2)
x_train2 = np.array(img_train2)

print(x_train2.shape)

print(y_train2.shape)

###view rotated images
fig = plt.figure(figsize = (15, 10))
for i in range(25):
    rand = random.randint(0,len(y_train2)-1)
    ax = plt.subplot(5, 5, i+1)
    
    img = x_train2[rand]
    if y_train2[rand] == 1:
        plt.title('B')
    elif y_train2[rand]==2:
        plt.title('M')
    else:
        plt.title('S')
    plt.tight_layout()
    plt.axis('off')
    plt.imshow(img)
plt.show()

###
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 180,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.6, # Randomly zoom image 
        width_shift_range=0.5,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.5,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(x_train)

###
x_train2=[cv2.resize(a,(rows2,cols2)) for a in x_train2]
y_train2 = np.array(y_train2)
x_train2 = np.array(x_train2)
x_test2=[cv2.resize(a,(rows2,cols2)) for a in x_test2]
x_test2 = np.array(x_test2)
y_test2=np.array(y_test2)
(a,b,c)=x_train2.shape
x_train2=np.reshape(x_train2,(a,b,c,1))
(a,b,c)=x_test2.shape
x_test2=np.reshape(x_test2,(a,b,c,1))
print(x_train2.shape)

### model : 
def model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224, 224, 1)))
    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

      
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


model2 = model()
model2.summary()

model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
early_stop = EarlyStopping(monitor='loss', mode='auto', patience=1,restore_best_weights=True, verbose=1)

check_point_filepath = 'D:/Cours CS/Projet S8/Project-S8-Mammography'

model_check_point = ModelCheckpoint(filepath =check_point_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')



hist = model2.fit(x_train2,
                 y_train2,
                 epochs=5,
                 batch_size=64,
                 callbacks=[early_stop, model_check_point],validation_data=(x_test2,y_test2),initial_epoch=0)




model2.evaluate(x_test2,y_test2)
model2.predict(x_test2)
y_test3=y_test2
for i in range(len(y_test3)):
    if y_test3[i]==2:
        y_test3[i]=1

model2.evaluate(x_test2,y_test3)
###
model3=model()

