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
          '5: Show Test Result in Graph\n\t'
          '9: Data Preprocessing\n\n')
    sys.exit(1)

cmd = int(sys.argv[1])
arg = int(sys.argv[2]) if cmd != 5 and cmd != 9 else 0

if (cmd != 5 and cmd != 9) and arg == 0:
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
        cnn.train(
            *data.load_training_dataset(),
            epoch=arg,
            test_result=data.load_test_result(epoch=arg - 1000) if arg != 1000 else None
        )

    # Test
    if cmd == 2:
        cnn = CNN()
        data = Data()
        cnn.test(*data.load_test_dataset(), epoch=arg)

    # Training Result
    if cmd == 3:
        data = Data()
        training_result = data.load_training_result(epoch=arg)
        for filename, result in training_result.items():
            print(filename)
            for k, v in result.items():
                print(k + ':', v)
            print()

    # Test Result
    if cmd == 4:
        data = Data()
        test_result = data.load_test_result(epoch=arg)
        for k, v in test_result.items():
            print(k + ':', str(v) + '%')

    # SHow Test Result in Graph
    if cmd == 5:
        data = Data()
        data.show_test_result()

    # Data Preprocessing
    if cmd == 9:
        data = Data()
        data.preprocess()
