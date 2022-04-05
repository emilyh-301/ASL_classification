# Emily Haigh and Anthony An
# American Sign Language to text
# Starting with A-Z

# Disable TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

# https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/
from cnn import CNN
from data import Data
from trace import Trace


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('\nSet the command number:\n\t1: Training\n\t2: Test\n\t9: Data Preprocessing\n\n')
        sys.exit(1)
    cmd = int(sys.argv[1])

    # Training
    if cmd == 1:
        Trace().start()
        cnn = CNN()
        data = Data()
        cnn.model(*data.load_training_dataset())

    # Test
    if cmd == 2:
        cnn = CNN()
        data = Data()
        cnn.load(*data.load_test_dataset())

    # Data Preprocessing
    if cmd == 9:
        data = Data()
        data.preprocess()
