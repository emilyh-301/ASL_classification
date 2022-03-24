# Emily Haigh and Anthony An
# American Sign Language to text
# Starting with A-Z

# Disable TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/

from cnn import CNN
from data import Data
from trace import Trace


if __name__ == '__main__':
    Trace().start()
    cnn = CNN()
    data = Data()
    cnn.model(*data.load())
    # cnn.load()
