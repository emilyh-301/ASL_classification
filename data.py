import csv
import math
import random

import numpy as np
from PIL import Image

from path import Path

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class Data(Path):
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
            cat_path = self.path('Dataset', category)
            training_path = self.path('Training', category + '.csv')
            test_path = self.path('Test', category + '.csv')

            training_dataset, test_dataset = self._split_training_test(self.listdir(cat_path))
            for img_filename in training_dataset:
                img_path = self.path('Dataset', category, img_filename)
                self._img_2_csv(img_path=img_path, csv_path=training_path)
            for img_filename in test_dataset:
                img_path = self.path('Dataset', category, img_filename)
                self._img_2_csv(img_path=img_path, csv_path=test_path)

        print('\nThe dataset has been preprocessed.\n')

    def load_training_dataset(self) -> tuple:
        dataset, labels = self._load_dataset(directory='Training')
        activations = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu', 'exponential']
        optimizers = ['sgd', 'rmsprop', 'adam', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl']
        losses = [
            'losses.SparseCategoricalCrossentropy(from_logits=True)',
            'losses.Poisson()',
            'losses.KLDivergence()',
            'losses.MeanSquaredError()',
            'losses.MeanAbsoluteError()',
            'losses.MeanAbsolutePercentageError()',
            'losses.MeanSquaredLogarithmicError()',
            'losses.Huber()',
            'losses.LogCosh()',
            'losses.Hinge()',
            'losses.SquaredHinge()',
            'losses.CategoricalHinge()'
        ]
        return dataset, labels, activations, optimizers, losses

    def load_test_dataset(self) -> tuple:
        dataset, labels = self._load_dataset(directory='Test')
        return dataset, labels

    def _load_dataset(self, directory) -> tuple:
        dataset = list()
        labels = list()

        # A=0 ... Z=25
        for category in categories:
            csv_path = self.path(directory, category + '.csv')
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
