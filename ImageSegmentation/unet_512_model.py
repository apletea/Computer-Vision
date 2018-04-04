# coding: utf-8
'''
    - "ZF_UNET_224" Model based on UNET code from following paper: https://arxiv.org/abs/1505.04597
    - This model used to get 2nd place in DSTL competition: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
    - For training used DICE coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    - Input shape for model is 224x224 (the same as for other popular CNNs like VGG or ResNet)
    - It has 3 input channels (to process standard RGB (BGR) images). You can change it with variable "INPUT_CHANNELS"
    - It trained on random image generator with random light shapes (ellipses) on dark background with noise (< 10%).
    - In most cases model ZF_UNET_224 is ok to be used without pretrained weights.
'''

__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Activation
from keras import backend as K
from keras.layers.merge import concatenate

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


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


def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = Dropout(dropout)(conv)
    return conv


def UNET_512(dropout_val=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 512, 512))
        axis = 1
    else:
        inputs = Input((512, 512, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_512 = double_conv_layer(inputs, filters, dropout_val, batch_norm)
    pool_256 = MaxPooling2D(pool_size=(2, 2))(conv_512)

    conv_256 = double_conv_layer(pool_256, 2*filters, dropout_val, batch_norm)
    pool_128 = MaxPooling2D(pool_size=(2, 2))(conv_256)

    conv_128 = double_conv_layer(pool_128, 4*filters, dropout_val, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2, 2))(conv_128)

    conv_64 = double_conv_layer(pool_64, 8*filters, dropout_val, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2, 2))(conv_64)

    conv_32 = double_conv_layer(pool_32, 16*filters, dropout_val, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2, 2))(conv_32)

    conv_16 = double_conv_layer(pool_16, 32*filters, dropout_val, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)

    conv_8 = double_conv_layer(pool_8,64*filters,dropout_val,batch_norm)

    up_16 = concatenate([UpSampling2D(size=(2, 2))(conv_8), conv_16], axis=axis)
    up_conv_16 = double_conv_layer(up_16, 16*filters, dropout_val, batch_norm)

    up_32 = concatenate([UpSampling2D(size=(2, 2))(up_conv_16), conv_32], axis=axis)
    up_conv_32 = double_conv_layer(up_32, 8*filters, dropout_val, batch_norm)

    up_64 = concatenate([UpSampling2D(size=(2, 2))(up_conv_32), conv_64], axis=axis)
    up_conv_64 = double_conv_layer(up_64, 4*filters, dropout_val, batch_norm)

    up_128 = concatenate([UpSampling2D(size=(2, 2))(up_conv_64), conv_128], axis=axis)
    up_conv_128 = double_conv_layer(up_128, 2*filters, dropout_val, batch_norm)

    up_256 = concatenate([UpSampling2D(size=(2, 2))(up_conv_128), conv_256], axis=axis)
    up_conv_256 = double_conv_layer(up_256, filters, 0, batch_norm)

    up_512 = concatenate([UpSampling2D(size=(2, 2))(up_conv_256), conv_512], axis=axis)
    up_conv_512 = double_conv_layer(up_512, filters, 0, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_512)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")
    return model

def ZF_UNET_512(dropout_val=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 512, 512))
        axis = 1
    else:
        inputs = Input((512, 512, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters, dropout_val, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, dropout_val, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, dropout_val, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, dropout_val, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters, dropout_val, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters, dropout_val, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters, dropout_val, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters, dropout_val, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, dropout_val, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, dropout_val, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, 0, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")
    return model

def ZF_UNET_224(dropout_val=0.0, batch_norm=True):
    if K.image_dim_ordering() == 'th':
        inputs = Input((INPUT_CHANNELS, 224, 224))
        axis = 1
    else:
        inputs = Input((224, 224, INPUT_CHANNELS))
        axis = 3
    filters = 32

    conv_224 = double_conv_layer(inputs, filters, dropout_val, batch_norm)
    pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)

    conv_112 = double_conv_layer(pool_112, 2*filters, dropout_val, batch_norm)
    pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)

    conv_56 = double_conv_layer(pool_56, 4*filters, dropout_val, batch_norm)
    pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)

    conv_28 = double_conv_layer(pool_28, 8*filters, dropout_val, batch_norm)
    pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)

    conv_14 = double_conv_layer(pool_14, 16*filters, dropout_val, batch_norm)
    pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)

    conv_7 = double_conv_layer(pool_7, 32*filters, dropout_val, batch_norm)

    up_14 = concatenate([UpSampling2D(size=(2, 2))(conv_7), conv_14], axis=axis)
    up_conv_14 = double_conv_layer(up_14, 16*filters, dropout_val, batch_norm)

    up_28 = concatenate([UpSampling2D(size=(2, 2))(up_conv_14), conv_28], axis=axis)
    up_conv_28 = double_conv_layer(up_28, 8*filters, dropout_val, batch_norm)

    up_56 = concatenate([UpSampling2D(size=(2, 2))(up_conv_28), conv_56], axis=axis)
    up_conv_56 = double_conv_layer(up_56, 4*filters, dropout_val, batch_norm)

    up_112 = concatenate([UpSampling2D(size=(2, 2))(up_conv_56), conv_112], axis=axis)
    up_conv_112 = double_conv_layer(up_112, 2*filters, dropout_val, batch_norm)

    up_224 = concatenate([UpSampling2D(size=(2, 2))(up_conv_112), conv_224], axis=axis)
    up_conv_224 = double_conv_layer(up_224, filters, 0, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_224)
    conv_final = BatchNormalization(axis=axis)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="ZF_UNET_224")
    return model
