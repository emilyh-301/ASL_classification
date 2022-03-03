# Emily Haigh and Anthony An
# American Sign Language to text
# Starting with A-Z
import os

import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from PIL import Image

# https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/

def load():
    categories = ['A']
    for category in categories:
        x_array = []
        y_array = []
        path = os.path.join('Dataset', category)  # create path to dogs and cats
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            y_array.append(category)
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (100, 100))
            x_array.append(img_array)
        pickle_out = open('x.pickle', "wb")
        pickle.dump(x_array, pickle_out)
        pickle_out.close()
        pickle_out = open('y.pickle', "wb")
        pickle.dump(y_array, pickle_out)
        pickle_out.close()

        break
    # img = Image.open('Dataset/A/A1000.jpg')
    # img = img.resize((100,100),Image.ANTIALIAS)
    # arr = np.array(img)  # 640x480x4 array
    #
    #
    # pickle_out = open('test.pickle', "wb")
    # pickle.dump(arr, pickle_out)
    # pickle_out.close()
    # arr[20, 30]  # 4-vector, just like above

    # with open('Dataset/A/A1000.jpg', 'rb') as f:
    #     file = f.read()
    # image = pickle.load(file)
    # print(image)
    # file.close()

def asl_text():
    #We have to change this to fit our data with testTrainSplit (X_train, Y_train), (X_test. Y_test) X contains data, Y contains labels
    X = pickle.load(open("x.pickle", "rb"))
    Y = pickle.load(open("y.pickle", "rb"))
    # normalizing
    # X = X / 255.0

    model = Sequential()

    # first layer
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layer
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third layer
    model.add(Flatten())
    model.add(Dense(64))

    # output layer
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="categorical", optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, batch_size=1, epochs=2, validation_split=.2)


if __name__ == '__main__':
    # load()
    asl_text()
    # pickle_in = open('test.pickle', 'rb')
    # example = pickle.load(pickle_in)
    # print(example)