# import matplotlib.pyplot as pp
import numpy as np
import tensorflow as tf

# Download CIFAR-100 dataset.
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()

# Check shapes.
assert train_x.shape == (50000, 32, 32, 3)
assert train_y.shape == (50000, 1)
assert test_x.shape == (10000, 32, 32, 3)
assert test_y.shape == (10000, 1)

# # Show a sample to check everything is fine.
# print(train_y[0])
# pp.imshow(train_x[0])
# pp.show()

# Reshape features for writing to file.
train_x = np.reshape(train_x, (-1, 32 * 32 * 3))
test_x = np.reshape(test_x, (-1, 32 * 32 * 3))

# Write dataset to file.
with open('data/cifar100', 'w') as f:
    for x, y in zip(train_x, train_y):
        f.write(",".join([str(y[0])] + [str(el) for el in x]) + "\n")
    for x, y in zip(test_x, test_y):
        f.write(",".join([str(y[0])] + [str(el) for el in x]) + "\n")
