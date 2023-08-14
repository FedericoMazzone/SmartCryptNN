import re
from ast import literal_eval

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def load_model(
    file_path
):
    ACTIVATION_FUNCTION = tf.nn.sigmoid

    # Read and parse dataset from file.
    with open(file_path) as f:
        lines = f.readlines()
    regex_weights = r"^w\d+="
    regex_biases = r"^b\d+="
    weights = [np.array(literal_eval(l[match.end():-1]))
               for l in lines if (match := re.match(regex_weights, l))]
    biases = [np.array(literal_eval(l[match.end():-1]))
              for l in lines if (match := re.match(regex_biases, l))]
    assert len(weights) == len(biases)
    assert len(weights) > 0

    # Turn into tensorflow model
    model = keras.Sequential(
        [
            keras.layers.Dense(
                weights[0].shape[1],
                activation=ACTIVATION_FUNCTION,
                input_shape=(weights[0].shape[0],)
            ),
        ]
    )
    for i in range(1, len(weights)):
        model.add(
            keras.layers.Dense(
                weights[i].shape[1],
                activation=ACTIVATION_FUNCTION
            )
        )
    model.set_weights([m for wb in zip(weights, biases) for m in wb])

    return model


if __name__ == "__main__":

    from load_mnist import load_MNIST

    model = load_model("data/models/plaintext_trained_model.txt")

    TRAIN_SIZE = 0
    TEST_SIZE = 100

    _, _, test_x, test_y = load_MNIST(
        file_path="data/mnist.txt",
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        to_categorical=True
    )

    model.compile(metrics="accuracy")
    model.evaluate(test_x, test_y)
