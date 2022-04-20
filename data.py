import csv
import math
import random
import json
import numpy as np
from PIL import Image

from path import Path

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class Data(Path):
    # Convert image files to csv files.
    def preprocess(self) -> None:
        # If directory does not exist, create one.
        if not self.exists('Training'):
            self.create('Training')
        if not self.exists('Test'):
            self.create('Test')

        # If directory is not empty, delete all files.
        if not self.is_empty('Training'):
            self.delete('Training')
            self.create('Training')
        if not self.is_empty('Test'):
            self.delete('Test')
            self.create('Test')

        for category in categories:
            print(category)
            cat_path = self.join('Dataset', category)
            training_path = self.join('Training', category + '.csv')
            test_path = self.join('Test', category + '.csv')

            training_dataset, test_dataset = self._split_training_test(self.listdir(cat_path))
            for img_filename in training_dataset:
                img_path = self.join('Dataset', category, img_filename)
                self._img_2_csv(img_path=img_path, csv_path=training_path)
            for img_filename in test_dataset:
                img_path = self.join('Dataset', category, img_filename)
                self._img_2_csv(img_path=img_path, csv_path=test_path)

        print('\nThe dataset has been preprocessed.\n')

    def load_training_dataset(self) -> tuple:
        dataset, labels = self._load_dataset(directory='Training')
        hidden_activations = ['relu', 'sigmoid', 'tanh']
        output_activations = ['linear', 'sigmoid', 'softmax']
        optimizers = ['sgd', 'rmsprop', 'adam', 'adadelta', 'adagrad']
        losses = ['losses.SparseCategoricalCrossentropy(from_logits=True)']
        return dataset, labels, hidden_activations, output_activations, optimizers, losses

    def load_training_result(self, epoch: int) -> None:
        filepath = self.join(self.var.history_dir, str(epoch))
        for filename in sorted(self.listdir(filepath)):
            d = json.load(open(self.join(filepath, filename)))
            print(filename)

            # keys = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
            for k, v in d.items():
                max_epoch = max([e for e in v.keys()])
                print(k, v[str(max_epoch)])
            print()

    def load_test_dataset(self) -> tuple:
        dataset, labels = self._load_dataset(directory='Test')
        return dataset, labels

    def load_test_result(self, epoch: int) -> None:
        content = self.read(filepath=self.join(self.var.result_dir, str(epoch) + '.txt'))
        content_dict = dict()
        for i in range(0, len(content), 2):
            content_dict[content[i]] = float(content[i+1].split(' ')[1][:-1])
        content_dict = dict(reversed(sorted(content_dict.items(), key=lambda item: item[1])))

        for k, v in content_dict.items():
            print(k + ':', str(v) + '%')

    def _load_dataset(self, directory) -> tuple:
        dataset = list()
        labels = list()

        # A=0 ... Z=25
        for category in categories:
            csv_path = self.join(directory, category + '.csv')
            cat_dataset = self._csv_2_list(csv_path=csv_path)
            cat_label = [[ord(category) - 65]] * len(cat_dataset)
            dataset.extend(cat_dataset)
            labels.extend(cat_label)
            print('\n' + category + ' has been loaded.\n')

        return np.array(dataset), np.array(labels)

    def _csv_2_list(self, csv_path: str, mode='r') -> list:
        images = list()
        with open(csv_path, mode) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                # Convert 1d array to 2d array
                pixel_count = int(math.sqrt(len(row)))
                image_2d = np.reshape(row, (-1, pixel_count))
                images.append(image_2d)
        return images

    def _img_2_csv(self, img_path: str, csv_path: str, mode='a') -> None:
        img = Image.open(img_path)
        img_arr = np.array(img).flatten()
        with open(csv_path, mode) as f:
            write = csv.writer(f)
            write.writerow(img_arr)

    def _split_training_test(self, dataset: np.array) -> tuple:
        random.shuffle(dataset)
        training = dataset[:int(len(dataset) * 0.8)]
        test = dataset[int(len(dataset) * 0.8):]
        return training, test
