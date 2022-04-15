# Emily Haigh and Anthony An
# American Sign Language to text
# Starting with A-Z
import sys

if len(sys.argv) < 2:
    print('\nSet the command number:\n\t'
          '1: Training\n\t'
          '2: Test\n\t'
          '3: Training Result\n\t'
          '4: Test Result\n\t'
          '9: Data Preprocessing\n\n')
    sys.exit(1)

cmd = int(sys.argv[1])
arg = int(sys.argv[2]) if cmd != 9 else 0

if cmd != 9 and arg == 0:
    print('Specify the epoch number.\n')
    sys.exit(1)


# Disable TF warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/
from cnn import CNN
from data import Data
from trace import Trace

if __name__ == '__main__':
    # Training
    if cmd == 1:
        Trace().start()
        cnn = CNN()
        data = Data()
        cnn.train(*data.load_training_dataset(), epoch=arg)

    # Test
    if cmd == 2:
        cnn = CNN()
        data = Data()
        cnn.test(*data.load_test_dataset(), epoch=arg)

    # Training Result
    if cmd == 3:
        data = Data()
        data.load_training_result(epoch=arg)

    # Test Result
    if cmd == 4:
        data = Data()
        data.load_test_result(epoch=arg)

    # Data Preprocessing
    if cmd == 9:
        data = Data()
        data.preprocess()
