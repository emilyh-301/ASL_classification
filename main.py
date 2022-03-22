# Emily Haigh and Anthony An
# American Sign Language to text
# Starting with A-Z

# https://pythonprogramming.net/loading-custom-data-deep-learning-python-tensorflow-keras/

from cnn import CNN
from data import Data


if __name__ == '__main__':
    cnn = CNN()
    data = Data()
    cnn.loop(*data.load())
