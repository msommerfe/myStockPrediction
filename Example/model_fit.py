import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import numpy as np
import time
from tensorflow.keras.callbacks import TensorBoard

NAME = "Cats-vs-dogs-CNN-{}".format(int(time.time()))

def fit_model_Cat_dog():
    X=np.load('features.npy')
    y=np.load('labels.npy')

    X = tf.keras.utils.normalize(X, axis = 1)

    model = Sequential()
    #model.add(Conv2D(254, (1,1), input_shape=X.shape[1:]))
    model.add(Conv2D(254, (3,3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size = (1,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Conv2D(254, (1, 1)))
    model.add(Conv2D(254, (3, 3)))
    model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])


def fit_model_hand_written_digits():

    mnist = tf.keras.datasets.mnist # 28*28 hand-written difits 0-9
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test  = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs = 10)

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

    model.save('myFristModel')
    nmodel = tf.keras.models.load_model('myFristModel')
    prediction = nmodel.predict([x_test])
    print(prediction[1])


fit_model_hand_written_digits()
