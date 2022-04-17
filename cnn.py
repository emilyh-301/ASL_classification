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

DEFAULT_EPOCH = 1000
BATCH_SIZE = 128

path = Path()
trace = Trace()


class CNN:
    def test(self, *data, epoch: int) -> None:
        x, y = data
        x = self._reshape_dataset(x=x)
        model_path = path.join(path.var.model_dir, str(epoch))
        names = path.listdir(model_path)

        for name in names:
            try:
                print(name, '\n')
                model = self._load_model(filepath=path.join(model_path, name))
                prediction = model.predict(x=x)
                accuracy = self._accuracy(prediction=prediction, y=y)
                path.write(
                    filepath=path.join(path.var.result_dir, str(epoch) + '.txt'),
                    content=name + '\n' + 'Accuracy: ' + str(accuracy) + '%' + '\n',
                    mode='a'
                )
            except Exception as e:
                print(e)
                print('\n\n\n')

    def train(self, *data, epoch: int) -> None:
        trace.update_liveness(alive=True)
        x, y, hidden_activations, output_activations, optimizers, losses = data
        x = self._reshape_dataset(x=x)
        x, y = self._shuffle_inputs(x=x, y=y)

        for hidden_activation in hidden_activations:
            for output_activation in output_activations:
                for optimizer in optimizers:
                    for loss in losses:
                        name = hidden_activation + '_' + output_activation + '_' + optimizer + '_' + loss.split('.')[1].split('(')[0]

                        # Previous epoch model does not exist.
                        if epoch != 1000 and not self._model_exists(name=name, epoch=epoch - 1000):
                            continue
                        # The model already exists.
                        if self._model_exists(name=name, epoch=epoch):
                            continue
                        # The model is being created by another process.
                        if self._name_was_used(name=name):
                            continue

                        try:
                            print(name, '\n')
                            self._model(
                                hidden_activation,
                                output_activation,
                                optimizer,
                                loss,
                                name=name,
                                epoch=epoch,
                                x=x,
                                y=y,
                            )
                        except Exception:
                            path.write(
                                filepath=path.join(path.var.exception_dir, name + str(epoch) + '.txt'),
                                content=traceback.format_exc()
                            )
        trace.update_liveness(alive=False)

    def _accuracy(self, prediction: np.array, y: np.array) -> float:
        count = 0
        for i in range(len(y)):
            if prediction.argmax(axis=-1)[i] == y[i][0]:
                count += 1
        accuracy = round(count / len(y) * 100, 3)
        return accuracy

    def _create_model(self, *func, x: np.array) -> models.Sequential:
        hidden_activation, output_activation, optimizer, loss = func
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
        return model

    def _load_model(self, filepath: str) -> models.Sequential:
        return models.load_model(filepath=filepath)

    def _model(self, *func, name: str, epoch: int, x: np.array, y: np.array) -> None:
        model = self._create_model(*func, x=x) if epoch == 1000 else self._load_model(filepath=path.join(path.var.model_dir, str(epoch - 1000), name))
        history = model.fit(x, y, batch_size=BATCH_SIZE, epochs=DEFAULT_EPOCH, validation_split=.2)
        self._save_history(
            filepath=path.join(path.var.history_dir, str(epoch), name + '.json'),
            history_df=pd.DataFrame(history.history)
        )
        model.save(path.join(path.var.model_dir, str(epoch), name))

    def _model_exists(self, name: str, epoch: int) -> bool:
        for model_name in path.listdir(path.join(path.var.model_dir, str(epoch))):
            if model_name == name:
                return True
        return False

    def _name_was_used(self, name: str) -> bool:
        with FileLock(path.var.used_lock):
            names = path.read(filepath=path.var.used_filepath)
            if name not in names:
                path.write(filepath=path.var.used_filepath, content=name + '\n')
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
        del temp
        x, y = np.array(x), np.array(y)
        return x, y

    def _save_history(self, filepath: str, history_df: pd.DataFrame) -> None:
        with open(filepath, mode='w') as f:
            history_df.to_json(f)
