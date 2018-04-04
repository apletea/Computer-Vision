import os
import cv2
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical


def load_data(path,word2int,size):
    X = []
    y = []
    for dir in os.listdir(path):
        print dir
        for img in os.listdir(path+'/' + dir):
            X.append(cv2.resize(cv2.imread(path+'/' + dir + '/' + img),(size,size)))
            y.append(word2int[dir])
    X = np.array(X, np.float32) / 255
    return X,y

def preprocces(X,y):
    # X = np.array(X,np.float32)/255
    y = to_categorical(y)
    return X,y

def get_word2int(path):
    labels = pd.read_csv(path)
    word2int = {labels['class'][j]: j for j in xrange(0, len(labels))}
    return word2int

