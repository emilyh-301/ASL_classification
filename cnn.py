import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

class CNN:
    def asl_text(self, x: np.array, y: np.array):
        # normalizing
        x = x / 255.0
        image_count = len(x)
        pixel_count = len(x[0])
        channel_count = 1
        x = x.reshape(image_count, pixel_count, pixel_count, channel_count)

        model = Sequential()

        # first layer
        model.add(Conv2D(64, (3, 3), input_shape=(pixel_count, pixel_count, channel_count)))
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

        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

        model.fit(x, y, batch_size=1, epochs=2, validation_split=.2)