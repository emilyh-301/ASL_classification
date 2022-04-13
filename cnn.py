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

BATCH_SIZE = 128
EPOCHS = 1000

p = Path()
trace = Trace()


class CNN:
    def load(self, x: np.array, y: np.array) -> None:
        x = self._reshape_dataset(x=x)
        model_path = self._latest_model_path()
        names = p.listdir(model_path)
        for name in names:
            try:
                print(name)
                model = self._load_model(filepath=p.path(model_path, name))
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

    def model(self, x: np.array, y: np.array, hidden_activations: list, output_activations: list, optimizers: list, losses: list) -> None:
        trace.update_liveness(alive=True)
        x = self._reshape_dataset(x=x)
        x, y = self._shuffle_inputs(x=x, y=y)

        for hidden_activation in hidden_activations:
            for output_activation in output_activations:
                for optimizer in optimizers:
                    for loss in losses:
                        name = hidden_activation + '_' + output_activation + '_' + optimizer + '_' + loss.split('.')[1].split('(')[0]
                        if not self._was_name_used(name=name):
                            try:
                                print('\n', name)
                                self._create_model(name=name, x=x, y=y, hidden_activation=hidden_activation, output_activation=output_activation, optimizer=optimizer, loss=loss)
                            except Exception:
                                p.write(
                                    filepath=p.path('Exceptions', name + '.txt'),
                                    content=traceback.format_exc()
                                )
        trace.update_liveness(alive=False)

    def _create_model(self, name: str, x: np.array, y: np.array, hidden_activation: str, output_activation: str, optimizer: str, loss: str) -> None:
        model = models.Sequential()

        # first layer
        model.add(Conv2D(32, (3, 3), activation=hidden_activation, input_shape=x.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # second layer
        model.add(Conv2D(64, (3, 3), activation=hidden_activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # third layer
        model.add(Conv2D(64, (3, 3), activation=hidden_activation))

        # output layer
        model.add(Flatten())
        model.add(Dense(64, activation=output_activation))
        model.add(Dense(26))  # 26 letters

        model.compile(
            optimizer=optimizer,
            loss=eval(loss),
            metrics=['accuracy']
        )

        history = model.fit(x, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=.2)
        self._save_history(name=name + '_' + str(EPOCHS), history_df=pd.DataFrame(history.history))
        model.save(filepath=p.var.model_dir + str(EPOCHS) + '/' + name + '/')

    def _latest_model_path(self):
        dirs = p.listdir(p.var.model_dir)
        max_epochs = max([int(d) for d in dirs])
        return p.path(p.var.model_dir, str(max_epochs))

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
        with FileLock(p.var.used_lock):
            names = p.read(filepath=p.var.used_filepath)
            if name not in names:
                p.write(filepath=p.var.used_filepath, content=name + '\n')
        return True if name in names else False

    def _save_history(self, name: str, history_df: pd.DataFrame) -> None:
        with open(p.var.history_dir + name + '.json', mode='w') as f:
            history_df.to_json(f)
