import numpy as np
from tensorflow.keras import utils


def load_CIFAR100(
    file_path="data/cifar100",
    train_size=50000,
    test_size=10000,
    to_categorical=False,
    normalize=True
):
    FEATURE_SHAPE = (32, 32, 3)
    NUM_CLASSES = 100
    OG_TRAIN_SIZE = 50000
    OG_TEST_SIZE = 10000

    if test_size is None:
        test_size = OG_TEST_SIZE

    assert train_size >= 0 and train_size <= OG_TRAIN_SIZE
    assert test_size >= 0 and test_size <= OG_TEST_SIZE

    # Read and parse dataset from file.
    with open(file_path) as f:
        lines = f.readlines()

    train_x = np.array([[int(c) for c in l.split(',')[1:]] for l in lines[:train_size]])
    train_y = np.array([int(l.split(',')[0]) for l in lines[:train_size]])
    test_x = np.array([[int(c) for c in l.split(',')[1:]] for l in lines[OG_TRAIN_SIZE:OG_TRAIN_SIZE+test_size]])
    test_y = np.array([int(l.split(',')[0]) for l in lines[OG_TRAIN_SIZE:OG_TRAIN_SIZE+test_size]])

    # Reshape features.
    train_x = train_x.reshape((-1,) + FEATURE_SHAPE)
    test_x = test_x.reshape((-1,) + FEATURE_SHAPE)

    # Check shapes.
    assert train_size == 0 or train_x.shape == ((train_size,) + FEATURE_SHAPE)
    assert train_size == 0 or train_y.shape == (train_size,)
    assert test_size == 0 or test_x.shape == ((test_size,) + FEATURE_SHAPE)
    assert test_size == 0 or test_y.shape == (test_size,)

    # Normalize features in [-1, 1].
    if normalize:
        train_x = (train_x / 255 - 0.5) * 2
        test_x = (test_x / 255 - 0.5) * 2

    # Turn labels to one-hot encoding to fit neural network output layer.
    if to_categorical:
        train_y = utils.to_categorical(train_y, NUM_CLASSES)
        test_y = utils.to_categorical(test_y, NUM_CLASSES)

    return train_x, train_y, test_x, test_y


if __name__ == "__main__":

    import matplotlib.pyplot as pp

    train_x, train_y, test_x, test_y = load_CIFAR100(
        file_path="data/cifar100",
        train_size=1000,
        test_size=1000,
        to_categorical=True
    )

    # Show a sample to check everything is fine.
    id = 0
    print("label", train_y[id])
    pp.imshow(train_x[id])
    pp.show()
