import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets
import numpy as np 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10   #111
from matplotlib import pyplot
from keras.utils import np_utils


print('Python version : ', sys.version)
print('Keras version : ', keras.__version__)

img_rows = 32   #111
img_cols = 32
num_classes=10


(x_train, y_train), (x_test, y_test) =cifar10.load_data()

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



epochs = 100
filename='checkpoint.h5'.format(epochs)
early_stopping=EarlyStopping(monitor='val_loss',mode='min',patience=15,verbose=1)                           #얼리스타핑
checkpoint=ModelCheckpoint(filename,monitor='val_loss',verbose=1,save_best_only=True,mode='auto')           #체크포인트



inputs=x_train.shape[1:]
output = layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation ='relu')(inputs)
output= layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(output)
output= layers.MaxPooling2D(pool_size=(2,2), padding='same', strides=1)(output)

output = layers.GlobalAveragePooling2D()(output)
output = layers.Dense(1000, activation = 'relu', name='fc" + str(i) + "')(output)
output = layers.Dense(10, activation = 'softmax', name='output')(output)

model = keras.Model(inputs = inputs, outputs = output)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer ='relu', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:',  score[0])
print('Test accuracy:', score[1])
