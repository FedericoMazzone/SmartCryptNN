from io import StringIO

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras import layers, models, optimizers, utils

LABELS_URL = "https://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"

with requests.get(LABELS_URL) as response:
    data = response.text
start_index = data.find("person")
data = data[start_index:]
df = pd.read_csv(StringIO(data), sep='\t')
# print(df.head())

lfw_people = fetch_lfw_people(
    color=True
)


def get_value(name, property):
    return int(df[property][df["person"] == name].mean() > 0)


def get_name(index):
    person_id = lfw_people.target[index]
    person_name = lfw_people.target_names[person_id]
    return person_name


def get_male(index):
    return get_value(
        get_name(index),
        "Male"
    )


def get_black(index):
    return get_value(
        get_name(index),
        "Black"
    )


NUM_SAMPLES = 13233
FEATURE_SHAPE = (62, 47, 3)
# FEATURE_SHAPE = (62, 47)
NUM_CLASSES = 2

y_male = np.array([get_male(index) for index in range(len(lfw_people.images))])
y_black = np.array([get_black(index)
                   for index in range(len(lfw_people.images))])

x = lfw_people.images

assert(x.shape == (NUM_SAMPLES, ) + FEATURE_SHAPE)
assert(y_male.shape == (NUM_SAMPLES,))
assert(y_black.shape == (NUM_SAMPLES,))

# x = x.reshape(NUM_SAMPLES, *FEATURE_SHAPE, 1)

# y_male = utils.to_categorical(y_male, NUM_CLASSES)
# y_black = utils.to_categorical(y_black, NUM_CLASSES)

train_size = int(0.5 * NUM_SAMPLES)
test_size = NUM_SAMPLES - train_size

train_x = x[:train_size]
test_x = x[train_size:train_size+test_size]

train_y_male = y_male[:train_size]
test_y_male = y_male[train_size:train_size+test_size]
print(f"{train_y_male.mean()=}")
print(f"{test_y_male.mean()=}")

OPTIMIZER = tf.keras.optimizers.Adam()
# OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=0.01)
LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# LOSS = tf.keras.losses.MeanSquaredError()


def create_MLP(
    architecture,
    activation_function=tf.nn.sigmoid,
    max_value=1.0
):
    model = tf.keras.Sequential()
    # Add input layers
    model.add(tf.keras.layers.Input(shape=(62*47*3,)))
    # Add hidden layers
    for layer_size in architecture:
        model.add(
            tf.keras.layers.Dense(
                layer_size,
                activation=activation_function,
                kernel_initializer=tf.keras.initializers.RandomUniform(
                    minval=-max_value, maxval=max_value, seed=None),
                bias_initializer=tf.keras.initializers.RandomUniform(
                    minval=-max_value, maxval=max_value, seed=None)
            )
        )
    # Add output layer
    model.add(
        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation=activation_function,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-max_value, maxval=max_value, seed=None),
            bias_initializer=tf.keras.initializers.RandomUniform(
                minval=-max_value, maxval=max_value, seed=None)
        )
    )
    # Compile model
    model.compile(
        optimizer=OPTIMIZER,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    return model


def create_CNN():
    model = models.Sequential()
    # Three spatial convolution layers with 32, 64, and 128 filters
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=FEATURE_SHAPE))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    # Two fully connected layers of size 256 and 2
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(2))
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model


# Print the model summary
# model = create_MLP([1024, 256, 64, 16])
model = create_CNN()
model.summary()
NUM_LAYERS = len(model.trainable_variables) // 2
print(f"Number of layers: {NUM_LAYERS}")

BATCH_SIZE = 32

model.fit(
    train_x, train_y_male,
    batch_size=BATCH_SIZE,
    epochs=100,
    validation_data=(test_x, test_y_male)
)


def get_black_batches(batch_size, value=1):
    common_length = min((y_black == 0).sum(), (y_black == 1).sum())
    black_x = x[y_black == value]
    black_y = y_male[y_black == value]
    black_x = black_x[:common_length]
    black_y = black_y[:common_length]
    num_chunks = len(black_x) // batch_size
    black_x = black_x[:(num_chunks) * batch_size]
    black_y = black_y[:(num_chunks) * batch_size]
    return zip(np.array_split(black_x, num_chunks), np.array_split(black_y, num_chunks))


def get_gradients(x, y, grad_indices=range(NUM_LAYERS)):
    x_tf = tf.constant(x, dtype=tf.float32)
    y_tf = tf.constant(y, dtype=tf.int32)
    # Create a GradientTape to compute gradients
    with tf.GradientTape(persistent=True) as tape:
        # Make a forward pass through the model
        logits = model(x_tf)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_tf, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    # Get the gradients for each trainable variable in the model
    gradients = tape.gradient(loss, model.trainable_variables)
    # Release the resources used by the tape
    del tape
    # 'gradients' now contains the gradients for each trainable variable in the model
    # for var, grad in zip(model.trainable_variables, gradients):
    #     print(var.name, grad)
    # return gradients
    selected_gradients = []
    [selected_gradients.extend([gradients[2*i], gradients[2*i+1]])
     for i in grad_indices]
    flattened_gradients = [tf.reshape(grad, shape=(-1,))
                           for grad in selected_gradients]
    return tf.concat(flattened_gradients, axis=0)


def attack(grad_indices=range(NUM_LAYERS), batch_size=BATCH_SIZE, num_rep=100):
    black_grads = np.array([get_gradients(bx, by, grad_indices)
                           for (bx, by) in get_black_batches(batch_size)])
    non_black_grads = np.array([get_gradients(nbx, nby, grad_indices) for (
        nbx, nby) in get_black_batches(batch_size, value=0)])
    assert len(black_grads) == len(non_black_grads)
    atk_train_size = int(0.5 * len(black_grads))
    atk_test_size = len(black_grads) - atk_train_size
    atk_train_x = tf.concat(
        (black_grads[:atk_train_size], non_black_grads[:atk_train_size]), axis=0)
    atk_test_x = tf.concat(
        (black_grads[atk_train_size:], non_black_grads[atk_train_size:]), axis=0)
    atk_train_y = tf.concat(
        (tf.ones(atk_train_size, dtype=bool), tf.zeros(atk_train_size, dtype=bool)), axis=0)
    atk_test_y = tf.concat(
        (tf.ones(atk_test_size, dtype=bool), tf.zeros(atk_test_size, dtype=bool)), axis=0)
    # pca = PCA(n_components=10)
    # atk_train_x_reduced = pca.fit_transform(atk_train_x)
    # atk_test_x_reduced = pca.transform(atk_test_x)
    # print(atk_train_x_reduced.shape)
    accuracies = []
    for _ in range(num_rep):
        # Create a Random Forest classifier with 50 trees
        random_forest = RandomForestClassifier(n_estimators=50, verbose=0)
        random_forest.fit(atk_train_x, atk_train_y)
        y_pred = random_forest.predict(atk_test_x)
        # random_forest.fit(atk_train_x_reduced, atk_train_y)
        # y_pred = random_forest.predict(atk_test_x_reduced)
        accuracy = accuracy_score(atk_test_y, y_pred)
        accuracies.append(accuracy)
    print(accuracies)
    acc_avg = np.array(accuracies).mean()
    print(acc_avg)
    return acc_avg


print("Attack all: ", attack(num_rep=100))

attack_acc_overall = []

for layer_to_attack in range(NUM_LAYERS):
    grad_indices = [layer_to_attack]
    acc_avg = attack(grad_indices, num_rep=100)
    attack_acc_overall.append(acc_avg)

print(attack_acc_overall)
