

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow.keras import layers
import copy
import random
import time
import sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10  
from matplotlib import pyplot
from keras.utils import np_utils


def make_rand(net_list):
  lis=list()
  re_seed=random.randint(1,4) 
  for i in range(re_seed):
    seed=random.randint(1,4) 
    if seed==1:
     im_output= layers.Conv2D(filters=64, kernel_size=[ks,ks], padding='same', activation=actF)(output)
    elif seed==2:
      im_output= layers.Dropout(rate=drop_out)(output)
    elif seed==3:
     im_output= layers.MaxPooling2D(pool_size=[ks, ks], padding='same', strides=1)(output)
    elif seed==4:
     im_output = layers.Activation(actF)(output)
    lis.append(im_output)
  return lis
start_time = time.time()
start_clock = time.clock()
lr = 0.6184999956130981
initW = 'None'
opt = keras.optimizers.Adagrad(learning_rate=lr)
actF = 'relu'
ks = 5
depth = 5
fc_layer = 3
drop_out = 0.28
byp = 1

img_rows = 32
img_cols = 32
num_classes=10
(x_train, y_train), (x_test, y_test) =cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
batch_size=128


filename='checkpoint.h5'.format(35)
early_stopping=EarlyStopping(monitor='val_loss',mode='min',patience=15,verbose=1)
checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True)
epochs =35

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = keras.Input(shape = x_train.shape[1:], name = 'input')
output = layers.Conv2D(filters = 64, kernel_size = [ks, ks], padding = 'same', activation = actF)(inputs)

net_list=list()
add_num=0
for _ in range(depth):
  a=make_rand(net_list)
  net_list.extend(a)
  if len(a)==1:r_num=0 
  else:r_num=random.randint(0,len(a)-1)                 
  output=a[r_num]   
  
  short_cut_dec=random.randint(1,5)     
  if (short_cut_dec<=byp) and len(net_list)>1:
    add_num=add_num+1
    for _ in range( random.randint(0,len(net_list)-1) ):
      a_layer_num=random.randint(0,len(net_list)-1)
      if a_layer_num!=r_num:
       c=layers.Add()([net_list[a_layer_num],output])
       output=c
       net_list.append(c)
output = layers.GlobalAveragePooling2D()(output)
output = layers.Dense(1000, activation = actF, name='fc0')(output)
dropout = layers.Dropout(rate=drop_out)(output)
output = layers.Dense(1000, activation = actF, name='fc1')(dropout)
dropout = layers.Dropout(rate=drop_out)(output)
output = layers.Dense(1000, activation = actF, name='fc2')(dropout)
dropout = layers.Dropout(rate=drop_out)(output)
output = layers.Dense(10, activation = 'softmax', name='output')(dropout)

model = keras.Model(inputs = inputs, outputs = output)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),callbacks=[checkpoint,early_stopping])

score = model.evaluate(x_test, y_test, verbose=0)
end_time = time.time()
end_clock = time.clock()
train_time = end_time - start_time
train_clock = end_clock - start_clock
print('Time to train (time) = ', train_time)
print('Time to train (clock) = ', train_clock)
print("Accuracy=", score[1], "genetic")
model.save('./saved/model_9.h5')
