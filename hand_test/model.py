from keras.layers import *
from keras.models import *

from keras.layers import *
from keras.models import *
from keras.applications import *


from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical

def get_model_firstFineTune(num_clases,size):
    model = VGG16(weights='imagenet',include_top=False,input_shape=(size,size,3))

    for layer in model.layers :
        layer.trainable = False
    x = model.input
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization(momentum = 0.95)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization(momentum = 0.95)(x)
    out = Dense(num_clases,activation='softmax')(x)

    res_model = Model(model.input,out)
    # model = Sequential()
    #
    # model.add(Conv2D(40,5,5,input_shape=(size,size,3)))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(BatchNormalization(momentum=0.95))
    #
    # model.add(Conv2D(80, 5, 5, border_mode="same"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(BatchNormalization(momentum=0.95))
    #
    # # Third CONV => RELU => POOL Layer
    # # Convolution -> ReLU Activation Function -> Pooling Layer
    # model.add(Conv2D(160, 5, 5, border_mode="same"))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(BatchNormalization(momentum=0.95))
    #
    # # FC => RELU layers
    # #  Fully Connected Layer -> ReLU Activation Function
    # model.add(Flatten())
    # model.add(Dense(32))
    # model.add(Activation("relu"))
    # model.add(BatchNormalization(momentum=0.95))
    # # Using Softmax Classifier for Linear Classification
    # model.add(Dense(num_clases))
    # model.add(Activation("softmax"))
    return res_model


def get_model_secondFineTune(num_clases,size):
    # model = VGG16(weights='imagenet', include_top=False, input_shape=(size, size, 3))
    # x = model.input
    # x = Flatten()(x)
    # x = Dense(32, activation='relu',trainable=False)(x)
    # x = BatchNormalization(momentum = 0.95)(x)
    # x = Dense(64, activation='relu', trainable=False)(x)
    # x = BatchNormalization(momentum = 0.95)(x)
    # out = Dense(num_clases, activation='softmax', trainable=False)(x)

    # res_model = Model(model.input, out)

    return get_model_firstFineTune(num_clases,size)


def get_model_final(num_clases,size):
    # model = VGG16(weights='imagenet', include_top=False, input_shape=(size, size, 3))
    # x = model.input
    # x = Flatten()(x)
    # x = Dense(32, activation='relu')(x)
    # x = BatchNormalization(momentum = 0.95)(x)
    # x = Dense(64, activation='relu')(x)
    # x = BatchNormalization(momentum = 0.95)(x)
    # out = Dense(num_clases, activation='softmax')(x)
    #
    # res_model = Model(model.input, out)

    return get_model_firstFineTune(num_clases,size)

#model1 = get_model_firstFineTune(6,50)
#model2 = get_model_secondFineTune(6)
#model3 = get_model_final(6)