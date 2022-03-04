import os
import csv
import numpy as np
from PIL import Image

class Data:
    img_size = (50, 50)

    def preprocess(self) -> None:
        categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        for category in categories:
            cat_path = os.path.join('Dataset', category)
            csv_path = os.path.join('Dataset', 'csv', category + '.csv')
            if os.path.exists(csv_path): os.remove(csv_path)

            count = 0
            for img_filename in os.listdir(cat_path):
                count += 1
                if count % 5 != 1: pass
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
            cat_label = [category] * len(cat_dataset)
            dataset.extend(cat_dataset)
            labels.append(cat_label)
            print('\n' + category + ' has been loaded.\n')
            break

        return np.array(dataset), np.array(dataset)

    def img_2_csv(self, img_path: str, csv_path: str, mode='a') -> None:
        img = Image.open(img_path).resize((50, 50)).convert('L')
        img_arr = np.array(img).flatten()
        with open(csv_path, mode) as f:
            write = csv.writer(f)
            write.writerow(img_arr)

    def csv_2_list(self, csv_path: str, mode='r') -> list:
        arr = list()
        with open(csv_path, mode) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                arr.append(row)
        return arr