'''
Created on 26 Jul 2020

@author: marco
'''

import tensorflow as tf
import DataSets
import numpy as np
import time


NAME = "StockData-seq-128-128-2-{}".format(int(time.time()))
MODEL=4

#Create Train and TestData
(x_train, y_train) = DataSets.get_csv_data(1500, 400, False)
(x_test , y_test) =  DataSets.get_csv_data(50, 400, True)

model = tf.keras.models.Sequential()

if MODEL == 1:
    #---------------Model 1 --------------------
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
elif MODEL ==2:
    #---------------Model 2 --------------------
    model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape= (x_train.shape[1],1)))
    model.add(tf.keras.layers.LSTM(50, return_sequences=False))
    model.add(tf.keras.layers.Dense(25,  activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
elif MODEL ==3:
    #---------------Model 3 --------------------
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
elif MODEL ==4:
    #---------------Model 4 --------------------

    model.add(tf.keras.layers.Conv2D(254, (1,1), input_shape=x_train.shape[1:], activation = tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size = (1,1)))

    model.add(tf.keras.layers.Conv2D(254, (1, 1),activation = tf.nn.relu))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 1)))

    model.add(tf.keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


#is logging the perfomance and can be view wirh tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs\{}".format(NAME))


model.fit(x_train, y_train,  epochs = 10, callbacks=[tensorboard])
val_loss, val_acc = model.evaluate(x_test, y_test)
#-----------Model is trained. from now on only debug info------------








print("Güte: Val_loss / Val_Acc")
print(val_loss, val_acc)

#model.save('myFristModel')
#nmodel = tf.keras.models.load_model('myFristModel')
prediction = model.predict([x_test])
print("Testdaten")
print(np.round(prediction[0],3), y_test[0])
print(np.round(prediction[1],3), y_test[1])
print(np.round(prediction[2],3), y_test[2])
print(np.round(prediction[3],3), y_test[3])
print(np.round(prediction[4],3), y_test[4])
print(np.round(prediction[5],3), y_test[5])
print(np.round(prediction[6],3), y_test[6])
print(np.round(prediction[7],3), y_test[7])
print(np.round(prediction[8],3), y_test[8])
print(np.round(prediction[9],3), y_test[9])

#print(np.argmax(prediction[0]), y_test[0])
