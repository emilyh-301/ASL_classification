import random
import numpy as np
import traceback
from filelock import FileLock
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from path import Path

# Overall CNN format
# https://www.tensorflow.org/tutorials/images/cnn

# The difference between training, validation, and test set
# https://stats.stackexchange.com/a/96869


p = Path()

class CNN:
    def loop(self, x: np.array, y: np.array, activations: list, optimizers: list, losses: list) -> None:
        for activation in activations:
            for optimizer in optimizers:
                for loss in losses:
                    name = activation + '_' + optimizer + '_' + loss.split('.')[1].split('(')[0]
                    if not self._is_name_used(name=name):
                        try:
                            print('\n', name)
                            self.model(name=name, x=x, y=y, activation=activation, optimizer=optimizer, loss=loss)
                        except Exception:
                            p.write(
                                filepath=p.path('Exceptions', name + '.txt'),
                                content=traceback.format_exc()
                            )

    def model(self, name: str, x: np.array, y: np.array, activation: str, optimizer: str, loss: str) -> None:
        x = self._reshape_dataset(x=x)
        x, y = self._shuffle_inputs(x=x, y=y)

        model = Sequential()

        # first layer
        model.add(Conv2D(32, (3, 3), activation=activation, input_shape=x.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second layer
        model.add(Conv2D(64, (3, 3), activation=activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # third layer
        model.add(Conv2D(64, (3, 3), activation=activation))

        # output layer
        model.add(Flatten())
        model.add(Dense(64, activation=activation))
        model.add(Dense(26))    # 26 letters

        model.compile(
            optimizer=optimizer,
            loss=eval(loss),
            metrics=['accuracy']
        )

        model_checkpoint = self._init_checkpoint(name=name)
        model.fit(x, y, batch_size=1, epochs=2000, validation_split=.2, callbacks=[model_checkpoint])

    def _init_checkpoint(self, name) -> ModelCheckpoint:
        model_checkpoint_callback = ModelCheckpoint(
            filepath='Models/' + name + '/',
            save_weights_only=True,
            save_freq='epoch',
            period=1,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        return model_checkpoint_callback

    def _is_name_used(self, name: str) -> bool:
        with FileLock('used'):
            names = p.read(filepath='used.txt')
            if name not in names:
                p.write(filepath='used.txt', content=name + '\n')
        return True if name in names else False

    def _reshape_dataset(self, x: np.array) -> np.array:
        # Normalizing
        x = x / 255.0

        # Add an image channel. We use a mono-channel
        image_count = len(x)
        pixel_count = len(x[0])
        channel_count = 1
        x = x.reshape(image_count, pixel_count, pixel_count, channel_count)

        return x

    # https://github.com/keras-team/keras/issues/4298#issuecomment-258947029
    def _shuffle_inputs(self, x: np.array, y: np.array) -> tuple:
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        x, y = np.array(x), np.array(y)
        return x, y