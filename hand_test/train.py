import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU' : 1, 'CPU':4})
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
import os
import data
import model

train_path = './Marcel-Train'
test_path = './Marcel-Test'
size = 128
classes = os.listdir(train_path)

word2int = data.get_word2int('labels.txt')

X,y = data.load_data(train_path,word2int,size)
X,y = data.preprocces(X,y)
X_test,y_test = data.load_data(test_path,word2int,size)
X_test,y_test = data.preprocces(X_test,y_test)

mobile = model.get_model_firstFineTune(y_test.shape[1],size)
sgd = SGD()
mobile.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
checkpoint_path = './weights2.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path,monitor='val_loss', verbose=1,period=1,save_best_only=True)
# mobile.load_weights('./weights2.hdf5')
mobile.fit(X,y,nb_epoch=30,batch_size=150,verbose=1,validation_data=(X_test,y_test), callbacks=[checkpoint])
mobile.save('model.hdf5')

mobile = model.get_model_secondFineTune(y_test.shape[1],size)
mobile.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
checkpoint_path = './weights2.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path,monitor='val_loss', verbose=1,period=1,save_best_only=True)
mobile.load_weights('./weights2.hdf5')
mobile.fit(X,y,nb_epoch=400,batch_size=150,verbose=1,validation_data=(X_test,y_test), callbacks=[checkpoint])
mobile.save('model.hdf5')

mobile = model.get_model_final(y_test.shape[1],size)
mobile.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
checkpoint_path = './weights2.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path,monitor='val_loss', verbose=1,period=1,save_best_only=True)
mobile.load_weights('./weights2.hdf5')
mobile.fit(X,y,nb_epoch=400,batch_size=150,verbose=1,validation_data=(X_test,y_test), callbacks=[checkpoint])
mobile.save('model.hdf5')