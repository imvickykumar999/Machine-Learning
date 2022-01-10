
# RUN on CMD
# >>>python training.py 12 # Here, 12 is epochs number's ./model/model_12.h5

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys, os

import data_creation as dc
# dc = data_creation()
# os.system('CLS')

try:
    epochs = int(sys.argv[1])
except:
    epochs = 12 # default epochs

model_file = f'./model/model_{epochs}.h5'

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model.fit(dc.train_data, dc.train_targets, epochs=epochs, batch_size=1)

test_loss, test_acc = model.evaluate(dc.test_data, dc.test_targets)
model.save(model_file)

# print('*'*50, end='\n\n')
print('Test accuracy:', test_acc)
print('Test Loss:', test_loss)
