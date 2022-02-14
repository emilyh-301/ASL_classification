# Emily Haigh and Anthony An
# American Sign Language to text
# Starting with A-Z

import tensorflow as tf
from tensorflow.keras.models import Sequentail
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle


def asl_text():
    #We have to change this to fit our data with testTrainSplit (X_train, Y_train), (X_test. Y_test) X contains data, Y contains labels
    X = pickle.load(open("X.pickle", "rb"))
    Y = pickle.load(open("Y.pickle", "rb"))
    # normalizing
    X = X / 255.0

    model = Sequentail()

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

    model.compile(loss="catigorical", optimizer='adam', metrics=['accuracy'])

    model.fit(X, Y, batch_size=1, epochs=2, validation_split=.2)


if __name__ == '__main__':
    asl_text()


