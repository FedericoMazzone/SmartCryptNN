# Download MNIST dataset from keras
from keras.datasets import mnist
import numpy as np
import cv2
from pathlib import Path
# import matplotlib.pyplot as pp

# Load MNIST dataset.
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# Check shapes.
assert train_x.shape == (60000, 28, 28)
assert train_y.shape == (60000,)
assert test_x.shape == (10000, 28, 28)
assert test_y.shape == (10000,)

# Resize features to lower dimensionality 28x28 -> 8x8.
INPUT_SHAPE = (8, 8)
INPUT_SIZE = INPUT_SHAPE[0] * INPUT_SHAPE[1]
train_x = np.array([cv2.resize(x, dsize=INPUT_SHAPE) for x in train_x])
test_x = np.array([cv2.resize(x, dsize=INPUT_SHAPE) for x in test_x])

# # Show a sample to check everything is fine.
# pp.imshow(train_x[0], cmap='gray')
# pp.savefig("sample.png")
# print(train_y[0])

# Normalize features to [0, 1]
train_x = train_x / 255
test_x = test_x / 255

# Turn features from grey-scale to binary black-white image.
train_x[train_x < 0.5] = 0
train_x[train_x >= 0.5] = 1
test_x[test_x < 0.5] = 0
test_x[test_x >= 0.5] = 1

# Reshape features to fit neural network input layer.
train_x = np.reshape(train_x, (-1, INPUT_SIZE))
test_x = np.reshape(test_x, (-1, INPUT_SIZE))

# Write dataset to file.
data_dir = Path("data/")
data_dir.mkdir(exist_ok=True)
with open(data_dir.joinpath("mnist"), "w") as f:
    for x, y in zip(train_x, train_y):
        f.write("".join(str(int(el)) for el in x) + str(y) + "\n")
    for x, y in zip(test_x, test_y):
        f.write("".join(str(int(el)) for el in x) + str(y) + "\n")
