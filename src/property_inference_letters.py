import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

os.environ['CUDA_VISIBLE_DEVICES'] = ''


def show_im(example):
    example = example.reshape(28, 28)
    example = np.rot90(example, 3)
    example = np.flip(example, axis=1)
    plt.imshow(example, cmap="gray", vmin=0, vmax=255)
    plt.show()


train_df = pd.read_csv("data/EMNIST/emnist-byclass-train.csv", header=None)
test_df = pd.read_csv("data/EMNIST/emnist-byclass-test.csv", header=None)

MAX_PER_LETTER = None


def get_balanced_letters_by_case(df, max_per_letter=None):
    if max_per_letter is None:
        max_per_letter = df[0].value_counts().min()
    upper_case = []
    for df_label in range(10, 36):
        upper_case.append(df[df[0] == df_label].values[:max_per_letter, 1:])
    lower_case = []
    for df_label in range(36, 62):
        lower_case.append(df[df[0] == df_label].values[:max_per_letter, 1:])
    return upper_case, lower_case


train_upper_case, train_lower_case = get_balanced_letters_by_case(
    train_df, MAX_PER_LETTER)
test_upper_case, test_lower_case = get_balanced_letters_by_case(
    test_df, MAX_PER_LETTER)


def to_dataset(data):
    x = []
    y = []
    for i in range(26):
        x.append(data[i])
        y.append(np.full(len(data[i]), i))
    x = np.concatenate(x)
    y = np.concatenate(y)
    y = tf.keras.utils.to_categorical(y, 26)
    return x, y


train_upper_case_x, train_upper_case_y = to_dataset(train_upper_case)
train_lower_case_x, train_lower_case_y = to_dataset(train_lower_case)
test_upper_case_x, test_upper_case_y = to_dataset(test_upper_case)
test_lower_case_x, test_lower_case_y = to_dataset(test_lower_case)

train_x = np.concatenate((train_upper_case_x, train_lower_case_x))
train_y = np.concatenate((train_upper_case_y, train_lower_case_y))
test_x = np.concatenate((test_upper_case_x, test_lower_case_x))
test_y = np.concatenate((test_upper_case_y, test_lower_case_y))


def create_MLP(
    architecture,
    activation_function=tf.nn.sigmoid,
    max_value=1.0
):
    model = tf.keras.Sequential()
    # Add input layers
    model.add(tf.keras.layers.Input(shape=(28*28,)))
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
            26,
            activation=activation_function,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-max_value, maxval=max_value, seed=None),
            bias_initializer=tf.keras.initializers.RandomUniform(
                minval=-max_value, maxval=max_value, seed=None)
        )
    )
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=30.0),
        loss=tf.keras.losses.MeanSquaredError(),
        # optimizer=tf.keras.optimizers.Adam(),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )
    return model


# Print the model summary
model = create_MLP([64, 64])
model.summary()
NUM_LAYERS = len(model.trainable_variables) // 2
print(f"Number of layers: {NUM_LAYERS}")

BATCH_SIZE = 100

model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=100,
    shuffle=True,
    validation_data=(test_x, test_y)
)


def get_batches(data_x, data_y, batch_size):
    num_chunks = len(data_x) // batch_size
    data_x = data_x[:(num_chunks) * batch_size]
    data_y = data_y[:(num_chunks) * batch_size]
    return zip(np.array_split(data_x, num_chunks), np.array_split(data_y, num_chunks))


def get_gradients(x, y, grad_indices=range(NUM_LAYERS)):
    x_tf = tf.constant(x, dtype=tf.float32)
    y_tf = tf.constant(y, dtype=tf.int32)
    # Create a GradientTape to compute gradients
    with tf.GradientTape(persistent=True) as tape:
        # Make a forward pass through the model
        logits = model(x_tf)
        # loss = tf.keras.losses.sparse_categorical_crossentropy(
        #     y_tf, logits, from_logits=True)
        loss = tf.keras.losses.mean_squared_error(y_tf, logits)
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
    upper_grads = np.array([get_gradients(bx, by, grad_indices)
                           for (bx, by) in get_batches(train_upper_case_x, train_upper_case_y, batch_size)])
    lower_grads = np.array([get_gradients(nbx, nby, grad_indices) for (
        nbx, nby) in get_batches(train_lower_case_x, train_lower_case_y, batch_size)])
    assert len(upper_grads) == len(lower_grads)
    atk_train_size = int(0.5 * len(upper_grads))
    atk_test_size = len(upper_grads) - atk_train_size
    atk_train_x = np.concatenate(
        (upper_grads[:atk_train_size], lower_grads[:atk_train_size]), axis=0)
    atk_test_x = np.concatenate(
        (upper_grads[atk_train_size:], lower_grads[atk_train_size:]), axis=0)
    atk_train_y = np.concatenate(
        (np.ones(atk_train_size, dtype=bool), np.zeros(atk_train_size, dtype=bool)), axis=0)
    atk_test_y = np.concatenate(
        (np.ones(atk_test_size, dtype=bool), np.zeros(atk_test_size, dtype=bool)), axis=0)
    permuted_indices = np.random.permutation(len(atk_train_x))
    atk_train_x = atk_train_x[permuted_indices]
    atk_train_y = atk_train_y[permuted_indices]
    print(atk_train_x.shape)
    print(atk_train_y.shape)
    print(atk_test_x.shape)
    print(atk_test_y.shape)
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


print("Attack all: ", attack(num_rep=3, batch_size=10))

attack_acc_overall = []

for layer_to_attack in range(NUM_LAYERS):
    grad_indices = [layer_to_attack]
    acc_avg = attack(grad_indices, num_rep=3, batch_size=10)
    attack_acc_overall.append(acc_avg)

print(attack_acc_overall)
