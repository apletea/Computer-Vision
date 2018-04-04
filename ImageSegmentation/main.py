import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':16})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras import backend as K
import os

# def set_keras_backend(backend):
#
#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend
#
# set_keras_backend("theano")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
import h5py

from unet_512_model import ZF_UNET_224, dice_coef_loss, dice_coef
from keras.optimizers import Adam


img = cv2.imread("/home/please/001.png",1)
cv2.imshow('o',img)
cv2.waitKey()

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():

    inputs  = Input((224,224,1))

    conv1 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64,3, activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = merge([drop4,up6],mode='concat',concat_axis=3)
    conv6 = Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = merge([conv3,up7], mode='concat', concat_axis=3)
    conv7 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge7)
    conv7 =  Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merger8 = merge([conv2,up8],mode='concat',concat_axis=3)
    conv8 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(merger8)
    conv8 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64,2,activation='relu',padding='same',kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = merge([conv1,up9],mode='concat',concat_axis=3)
    conv9 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(1,1,activation='sigmoid')(conv9)
    model = Model(input = inputs,outputs=conv10)

    model.compile(optimizer=sgd(lr = 0.001), loss=dice_coef_loss, metrics=['accuracy',dice_coef])

    return model


def load_data():
    file_names = pd.read_csv('test.txt',header=None)
    X=[]
    y=[]
    gray = []
    i = 0
    for file in file_names[0]:
        X.append(cv2.resize(cv2.imread('./train/{}'.format(file),1),(224,224)))
        gray.append(cv2.resize(cv2.imread('./train/{}'.format(file), 0), (224, 224)))
        # y.append(cv2.resize(cv2.imread('./train_mask/{}'.format(file),0),(224,224)))
    X = np.array(X,np.float32)/255
    # y = np.array(y,np.float32)/255
    # y = np.reshape(y,(y.shape[0],y.shape[1],y.shape[2],1))
    print X.shape
    return X,y,gray






model = ZF_UNET_224()
model.load_weights('unet.hdf5')
# model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
# img_train,img_mask,_ = load_data()
# #
# optim = Adam()
# model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])
# history = model.fit(img_train,img_mask,batch_size=4,nb_epoch=150,verbose=1,validation_split=0.1,callbacks=[model_checkpoint])
#
# model.save_weights('after_train.hdf5')
#

# model = ZF_UNET_224()
# model.load_weights("zf_unet_224.h5") # optional
# optim = Adam()
# model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])
#
# history = model.fit_generator(img_train,img_mask,batch_size=4,nb_epoch=400,verbose=1,validation_split=0.1,callbacks=[model_checkpoint])
# pd.DataFrame(history.history).to_csv('zf_unet_224_train.csv', index=False)
data,res,X = load_data()



y_net = model.predict(data, verbose=1)
print y_net
cv2.namedWindow('net',0)
cv2.namedWindow('img',0)
for i in xrange(0,len(data)):

    # imgNet = cv2.bitwise_or(X[i],y_net[i])
    # imgOr = cv2.bitwise_or(X[i],res[i])
    # cv2.imshow('net output',imgNet)
    # cv2.imshow('res',imgOr)
    cv2.imshow('net',y_net[i] )
    cv2.imshow('img',data[i])
    cv2.waitKey()