import numpy as np
from tensorflow.keras import utils


def load_MNIST(
    file_path="data/mnist.txt",
    train_size=60000,
    test_size=10000,
    to_categorical=False,
    normalize=False
):
    NUM_FEATURES = 8 * 8
    NUM_CLASSES = 10
    OG_TRAIN_SIZE = 60000
    OG_TEST_SIZE = 10000

    if test_size is None:
        test_size = OG_TEST_SIZE

    assert train_size >= 0 and train_size <= OG_TRAIN_SIZE
    assert test_size >= 0 and test_size <= OG_TEST_SIZE

    # Read and parse dataset from file.
    with open(file_path) as f:
        lines = f.readlines()
    x = [[int(c) for c in l[:NUM_FEATURES]] for l in lines]
    y = [int(l[NUM_FEATURES]) for l in lines]
    train_x = np.array(x[:train_size])
    train_y = np.array(y[:train_size])
    test_x = np.array(x[OG_TRAIN_SIZE:OG_TRAIN_SIZE+test_size])
    test_y = np.array(y[OG_TRAIN_SIZE:OG_TRAIN_SIZE+test_size])

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

    import matplotlib.pyplot as pp

    train_x, train_y, test_x, test_y = load_MNIST(
        file_path="data/mnist.txt",
        train_size=100,
        test_size=100,
        to_categorical=True
    )

    # Show a sample to check everything is fine.
    id = 0
    print("label", train_y[id])
    pp.imshow(train_x[id].reshape((8, 8)), cmap='gray')
    pp.show()
