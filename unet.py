import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import cv2
import re
import keras
import tensorflow as tf
import math
import random

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

def get_random_crop(image, crop_height, crop_width,x,y):



    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## defining a frame for image and mask storage
train = {'img' : [],
           'mask' : []
          }

test = {'img' : [],
           'mask' : []
          }

## defining data Loader function
def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 256):
    imgNames = os.listdir(imgPath)
    maskNames = []
    
    ## generating mask names
    for mem in imgNames:
        maskNames.append(re.sub('\.jpg', '.png', mem))
    
    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'
    
    for i in tqdm.tqdm(range(len(imgNames))):
        try:
            img = cv2.imread(imgAddr + imgNames[i])
            mask = cv2.imread(maskAddr + maskNames[i])
            
        except:
            continue
        
        h,w = img.shape[0:2]

        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))
            
        frameObj['img'].append(img)
        frameObj['mask'].append(mask)
        
    return frameObj
        
    
train = LoadData( train, imgPath = 'datasets/images', 
                        maskPath = 'datasets/labels/pixel_level_labels_colored'
                         , shape = 256)

# Check if training data looks all right
ix = random.randint(0, 1000)
plt.imshow(train['img'][ix])
plt.show()
plt.imshow((train['mask'][ix]))
plt.show()

#%%

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred = tf.cast(y_pred, tf.int32)
        y_pred_ = y_pred > t
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(3, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)



model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['acc'])
model.load_weights('tf_unet_mod.h5')
model.summary()

#%%
def predict (valMap, model, shape = 256):
    ## getting and proccessing val data
    img = valMap['img'][2:3]
    mask = valMap['mask'][2:3]
    
    imgProc = img
    imgProc = np.array(img)
    
    predictions = model.predict(imgProc)
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i,:,:,0],predictions[i,:,:,1],predictions[i,:,:,2]))
    
    return predictions[0], imgProc[0], mask[0]

def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(7,7))
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('image')
    
    plt.subplot(1,3,2)
    plt.imshow(predMask)
    plt.title('Predicted Mask')
    
    plt.subplot(1,3,3)
    plt.imshow(groundTruth)
    plt.title('actual Mask')
    plt.show()


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    Prediction, actuals, masks = predict(train, model)
    Plotter(actuals, Prediction, masks)
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model--1-unet.h5', verbose=1, save_best_only=True)
results = model.fit(np.array(train['img']),
                    np.array(train['mask']),
                    validation_split=0.25,
                    batch_size=16,
                    epochs=200,
                    verbose=1,
                    callbacks=[earlystopper, checkpointer, DisplayCallback()])

#%%

plt.plot(results.history['loss'], label = 'training_loss')
plt.plot(results.history['acc'], label = 'training_accuracy')
plt.legend()
plt.grid(True)

plt.plot(results.history['val_loss'], label = 'Validation_loss')
plt.plot(results.history['val_acc'], label = 'Validation_accuracy')
plt.legend()
plt.grid(True)

model.save_weights('tf_unet_mod.h5')

#%%
def loadImage(imgPath, shape = 256):

    img = plt.imread(imgPath)
    img = cv2.resize(img, (shape, shape))

    return img

def predict_test(img, model, shape = 256):    
    img = img
    imgProc = np.array(img)
    imgProc = tf.expand_dims(img, axis=0)
    
    predictions = model.predict(imgProc)
    for i in range(len(predictions)):
        predictions[i] = cv2.merge((predictions[i,:,:,0],predictions[i,:,:,1],predictions[i,:,:,2]))
    predictions = tf.squeeze(predictions)
    
    return predictions, img

def Plotter_test(img, predMask):
    plt.figure(figsize=(7,7))
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('image')
    
    plt.subplot(1,3,2)
    plt.imshow(predMask)
    plt.title('Predicted Mask')
    plt.show()
    
img = loadImage('datasets/images/0010.jpg')


    
Prediction, actuals = predict_test(img, model)
Prediction = np.array(Prediction)

Plotter_test(actuals, Prediction)

def num(i):
    i = str(i)
    l = len(i)
    if l<4:
        return (4-l)*'0'+i
    return i
#%%
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    frame = cv2.resize(frame, (256,256))
    frame,_ = predict_test(frame,model) 
    # Display the resulting frame
    frame = np.array(frame)
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
#%%
import warnings

warnings.filterwarnings('ignore')
for i in tqdm.tqdm(range(1005,2099)):
    img = loadImage('datasets/test/{}.jpg'.format(num(i)))
    Prediction, actuals = predict_test(img, model)
    plt.figure(figsize=(7,7))
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title('image')
    
    plt.subplot(1,3,2)
    plt.imshow(Prediction)
    plt.title('Predicted Mask')
    plt.savefig('datasets/test_labels/{}.jpg'.format(num(i)))
    plt.show()

#img  = cv2.imread('E:/Python projects/Electrothon/1.png')
