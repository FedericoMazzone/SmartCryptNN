import numpy as np
from tensorflow.keras import utils


def load_texas100(
    file_path="data/texas100",
    train_size=10000,
    test_size=50000,
    to_categorical=False,
    normalize=False
):
    NUM_SAMPLE = 67330
    NUM_FEATURES = 6169
    NUM_CLASSES = 100

    if test_size is None:
        test_size = NUM_SAMPLE - train_size

    assert train_size >= 0
    assert test_size >= 0
    assert train_size + test_size <= NUM_SAMPLE

    # Read and parse dataset from file.
    with open(file_path) as f:
        # lines = f.readlines()
        lines = [next(f) for _ in range(train_size + test_size)]
    x = [[int(c) for c in l.split(',')[1:]] for l in lines]
    y = [int(l.split(',')[0]) for l in lines]
    train_x = np.array(x[:train_size])
    train_y = np.array(y[:train_size])
    test_x = np.array(x[train_size:train_size+test_size])
    test_y = np.array(y[train_size:train_size+test_size])

    # Check shapes.
    assert train_size == 0 or train_x.shape == (train_size, NUM_FEATURES)
    assert train_size == 0 or train_y.shape == (train_size,)
    assert test_size == 0 or test_x.shape == (test_size, NUM_FEATURES)
    assert test_size == 0 or test_y.shape == (test_size,)

    # Turn labels to one-hot encoding to fit neural network output layer.
    if to_categorical:
        train_y = utils.to_categorical(train_y, NUM_CLASSES)
        test_y = utils.to_categorical(test_y, NUM_CLASSES)

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":

    train_x, train_y, test_x, test_y = load_texas100(
        file_path="data/texas100",
        train_size=100,
        test_size=100,
        to_categorical=True
    )
