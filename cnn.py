import random
import numpy as np
import traceback
import pandas as pd
from filelock import FileLock
from tensorflow.keras import losses, models
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from path import Path
from trace import Trace

# Overall CNN format
# https://www.tensorflow.org/tutorials/images/cnn

# The difference between training, validation, and test set
# https://stats.stackexchange.com/a/96869

# Save and load model
# https://www.tensorflow.org/guide/keras/save_and_serialize

# Save and load model with checkpoint
# https://keras.io/api/callbacks/model_checkpoint/


path = Path()
trace = Trace()


class CNN:
    def load(self, x: np.array, y: np.array) -> None:
        x = self._reshape_dataset(x=x)
        names = path.listdir(path.var.model_dir)
        for name in names:
            try:
                print(name)
                model = self._load_model(filepath=path.var.model_dir + name + '/')
                prediction = model.predict(x=x)

                count = 0
                for i in range(len(y)):
                    if prediction.argmax(axis=-1)[i] == y[i][0]:
                        count += 1
                accuracy = round(count / len(y) * 100, 3)
                print('Accuracy:', str(accuracy) + '%')
                print()
            except Exception as e:
                print(e)
                print('\n\n\n')

    def model(self, x: np.array, y: np.array, activations: list, optimizers: list, losses: list) -> None:
        trace.update_liveness(alive=True)
        x = self._reshape_dataset(x=x)
        x, y = self._shuffle_inputs(x=x, y=y)

        for activation in activations:
            for optimizer in optimizers:
                for loss in losses:
                    name = activation + '_' + optimizer + '_' + loss.split('.')[1].split('(')[0]
                    if not self._was_name_used(name=name):
                        try:
                            print('\n', name)
                            self._create_model(name=name, x=x, y=y, activation=activation, optimizer=optimizer, loss=loss)
                        except Exception:
                            path.write(
                                filepath=path.path('Exceptions', name + '.txt'),
                                content=traceback.format_exc()
                            )
        trace.update_liveness(alive=False)

    def _create_model(self, name: str, x: np.array, y: np.array, activation: str, optimizer: str, loss: str) -> None:
        model = models.Sequential()

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
        model.add(Dense(26))  # 26 letters

        model.compile(
            optimizer=optimizer,
            loss=eval(loss),
            metrics=['accuracy']
        )

        history = model.fit(x, y, batch_size=32, epochs=1000, validation_split=.2)
        self._save_history(name=name, history_df=pd.DataFrame(history.history))
        model.save(filepath=path.var.model_dir + name + '/')

    def _load_model(self, filepath: str) -> models.Sequential:
        return models.load_model(filepath=filepath)

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
        del temp
        x, y = np.array(x), np.array(y)
        return x, y

    def _was_name_used(self, name: str) -> bool:
        with FileLock(path.var.used_lock):
            names = path.read(filepath=path.var.used_filepath)
            if name not in names:
                path.write(filepath=path.var.used_filepath, content=name + '\n')
        return True if name in names else False

    def _save_history(self, name: str, history_df: pd.DataFrame) -> None:
        with open(path.var.history_dir + name + '.json', mode='w') as f:
            history_df.to_json(f)
