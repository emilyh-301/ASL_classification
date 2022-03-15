import os
import shutil
import csv
import math
import numpy as np
from PIL import Image

class Data:
    def preprocess(self) -> None:
        categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        # If the csv folder does not exist, create one.
        if not os.path.exists(os.path.join('Dataset', 'csv')):
            os.mkdir(os.path.join('Dataset', 'csv'))

        # If the csv folder is not empty, delete all files.
        if len(os.listdir(os.path.join('Dataset', 'csv'))) != 0:
            shutil.rmtree(os.path.join('Dataset', 'csv'))
            os.mkdir(os.path.join('Dataset', 'csv'))
        
        for category in categories:
            cat_path = os.path.join('Dataset', category)
            csv_path = os.path.join('Dataset', 'csv', category + '.csv')
            if os.path.exists(csv_path): os.remove(csv_path)

            for img_filename in os.listdir(cat_path):
                img_path = os.path.join('Dataset', category, img_filename)
                self.img_2_csv(img_path=img_path, csv_path=csv_path)

        print('\nThe dataset has been preprocessed.\n')

    def load(self) -> tuple:
        categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        dataset = list()
        labels = list()

        for category in categories:
            csv_path = os.path.join('Dataset', 'csv', category + '.csv')
            cat_dataset = self.csv_2_list(csv_path=csv_path)
            cat_label = [[ord(category)]] * len(cat_dataset)
            dataset.extend(cat_dataset)
            labels.extend(cat_label)
            print('\n' + category + ' has been loaded.\n')

        return np.array(dataset), np.array(labels)

    def img_2_csv(self, img_path: str, csv_path: str, mode='a') -> None:
        img = Image.open(img_path)
        img_arr = np.array(img).flatten()
        with open(csv_path, mode) as f:
            write = csv.writer(f)
            write.writerow(img_arr)

    def csv_2_list(self, csv_path: str, mode='r') -> list:
        images = list()
        with open(csv_path, mode) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                pixel_count = int(math.sqrt(len(row)))
                image_2d = np.reshape(row, (-1, pixel_count))
                images.append(image_2d)
        return images
