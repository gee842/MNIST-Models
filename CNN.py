import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.utils import to_categorical


from sklearn.preprocessing import OneHotEncoder

batch_size = 64
num_classes = 10
epochs = 60


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape( x_train.shape[0],28,28,1)
x_test = x_test.reshape( x_test.shape[0],28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.add(Dense(512, input_shape=(784, ), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(256, input_shape=(512, )))
# model.add(LeakyReLU(alpha=0.05))
# model.add(Dropout(0.3))
# model.add(Dense(512, input_shape=(256, )))
# model.add(LeakyReLU(alpha=0.05))
# model.add(Dense(256, input_shape=(512, ), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(128, input_shape=(256, ), activation='relu'))
# model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val','t_loss','val_loss'], loc='upper left')
plt.show()


def plotimage(data):
    data = data * 255
    mat = data.reshape(28, 28)
    plt.imshow(mat, cmap='gray')
    plt.show()