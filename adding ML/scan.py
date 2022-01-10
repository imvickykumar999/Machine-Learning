
# import tensorflow as tf
# from keras.models import Sequential
# from keras.preprocessing.image import load_img
# from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
#
# def train():
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#     x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#     x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#     input_shape = (28, 28, 1)
#
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255
#
#     model = Sequential()
#     model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#     model.add(Dense(128, activation=tf.nn.relu))
#     model.add(Dropout(0.2))
#     model.add(Dense(10,activation=tf.nn.softmax))
#
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     model.fit(x=x_train,y=y_train, epochs= 12)
#     model.evaluate(x_test, y_test)
#     model.save("./vicks/model.h5")

import os, cv2
from keras.models import load_model

def test():
    url = 0
    model = load_model('./vicks/model.h5')
    directory = './uploads/mnistvix'
    lst, filst = [], []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):

            photo = os.path.join(directory, filename)
            image = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)

            image = cv2.resize(image, (28,28))
            image = 255-image
            pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)

            print('='*60)
            print('\t>>> Predicted Digit : ', pred.argmax())
            lst.append(pred.argmax())
            filst.append(filename)

        else:
            continue
    return lst, filst

# test()
