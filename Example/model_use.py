import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist # 28*28 hand-written difits 0-9
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test  = tf.keras.utils.normalize(x_test, axis=1)


model = tf.keras.models.load_model('myFristModel')
val_loss, val_acc = model.evaluate(x_test, y_test)
#print(val_loss, val_acc)
print('-----------')

predictions = model.predict([x_test])

print(np.argmax(predictions[1]))
