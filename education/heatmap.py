import pandas as pd
import keras
import cv2
from tqdm import tqdm
import cv2
import glob
import os
import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.python.framework import ops
from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
import h5py
from keras.optimizers import SGD


def global_average_pooling(x):
    return K.mean(x, axis=(2, 3))


def global_average_pooling_shape(input_shape):
    return input_shape[0:2]


def VGG16_convolutions():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=( None, None, 3)))
    model.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 5, 5, activation='relu', name='conv1_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 5, 5, activation='relu', name='conv2_1'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(384, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(192, 3, 3, activation='relu', name='conv4_3'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(192, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv5_3_1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name="conv5_3"))
    return model


def get_model():
    model = VGG16_convolutions()
    # model = load_model_weights(model, "./tmp/weights.hdf5")
    # model.add(Lambda(global_average_pooling,
    #                  output_shape=global_average_pooling_shape))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model


def load_model_weights(model, weights_path):
    print 'Loading model.'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = False
    f.close()
    print 'Model loaded.'
    return model


def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

# def load_inria_person():
#     data = pd.read_csv('copter_trainval.txt',header=None,sep=' ')
#     y = []
#     x = []
#     pos
#     for i in tqdm(xrange(0,len(data))):
#         img = cv2.imread('./JPEGImages/{}.jpg'.format(data[0][i]))
#         x.append(cv2.resize(img,(300,300)))
#         y.append([data[1][i]^1,data[1][i]&1])
#     print y
#     x = np.array(x,np.float32)/255
#     y = np.array(y,np.uint8)
#     return x,y


def load_inria_person():
    pos_data =  pd.read_csv('pos_ex.txt',header=None,sep=' ')
    neg_data =  pd.read_csv('neg_ex.txt',header=None,sep=' ')
    pos_images = []
    neg_images = []
    for i in tqdm(xrange(0,len(pos_data))):
        img = cv2.imread('./JPEGImages/{}.jpg'.format(pos_data[0][i]))
        pos_images.append(cv2.resize(img,(300,300)))
    for i in tqdm(xrange(0, len(neg_images))):
        img = cv2.imread('./JPEGImages/{}.jpg'.format(neg_data[0][i]))
        neg_images.append(cv2.resize(img, (300, 300)))
    pos_images = [np.transpose(img, (2, 0, 1)) for img in pos_images]
    neg_images = [np.transpose(img, (2, 0, 1)) for img in neg_images]
    y = [1] * len(pos_images) + [0] * len(neg_images)
    y = to_categorical(y, 2)
    X = np.float32(pos_images + neg_images)
    return X, y

def train():
    model = get_model()
    X, y = load_inria_person()
    print "Training.."
    checkpoint_path = "./tmp/weights.hdf5"
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, period=1)
    model.fit(X, y, nb_epoch=40, batch_size=32, validation_split=0.2, verbose=1, callbacks=[checkpoint])
    model.save('./tmp/model.hdf5')


def visualize_class_activation_map(model, original_img):
    width, height, _ = original_img.shape

    # Reshape to the network input shape (3, w, h).
    img = np.array([original_img])

    # Get the 512 input weights to the softmax.
    class_weights = model.layers[-1].get_weights()[0]
    final_conv_layer = get_output_layer(model, "conv5_3")
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    conv_outputs = conv_outputs[0, :, :, :]

    # Create the class activation map.
    cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
    for i, w in enumerate(class_weights[:, 1]):
        if (i >= 87):
            continue
        cam += w * conv_outputs[i, :, :]
    print "predictions", predictions
    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # heatmap[np.where(cam < 0.2)] = 0
    # print heatmap
    img = heatmap * 0.5 + original_img
    print img[0]
    cv2.addWeighted(original_img,0.7,heatmap,0.3,img)

    return img,heatmap

# train()


vc = cv2.VideoCapture('/home/please/work/_Video/ULTIMATE.mp4')
model = load_model('./tmp/model.hdf5')
while True:
    _, img = vc.read()
    img2,he = visualize_class_activation_map(model,img)
    cv2.imshow('or',img)
    cv2.imshow('map',img2)
    cv2.imshow('heatmap',he)
    cv2.waitKey(5)
