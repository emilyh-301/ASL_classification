import os
import shutil
import csv
import math
import numpy as np
from PIL import Image
from path import Path

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


class Data(Path):
    def preprocess(self) -> None:
        # If the csv folder does not exist, create one.
        if not self.exists('Dataset', 'csv'):
            self.create('Dataset', 'csv')

        # If the csv folder is not empty, delete all files.
        if not self.is_empty('Dataset', 'csv'):
            self.delete('Dataset', 'csv')
            self.create('Dataset', 'csv')

        for category in categories:
            cat_path = self.path('Dataset', category)
            csv_path = self.path('Dataset', 'csv', category + '.csv')
            if self.exists(csv_path):
                self.delete(csv_path)

            for img_filename in self.listdir(cat_path):
                img_path = self.path('Dataset', category, img_filename)
                self._img_2_csv(img_path=img_path, csv_path=csv_path)

        print('\nThe dataset has been preprocessed.\n')

    def load(self) -> tuple:
        dataset, labels = self._load_data()
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

    def _load_data(self) -> tuple:
        dataset = list()
        labels = list()

        for category in categories:
            csv_path = self.path('Dataset', 'csv', category + '.csv')
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
