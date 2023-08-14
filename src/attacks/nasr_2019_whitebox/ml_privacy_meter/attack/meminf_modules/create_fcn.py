import tensorflow as tf

keraslayers = tf.keras.layers


def fcn_module(inputsize, layer_size=128):
    """
    Creates a FCN submodule. Used in different attack components.
    Args:
    ------
    inputsize: size of the input layer
    """

    # FCN module
    fcn = tf.keras.Sequential(
        [
            keraslayers.Dense(
                layer_size,
                activation=tf.nn.relu,
                input_shape=(inputsize,),
                kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
                bias_initializer='zeros'
            ),
            keraslayers.Dense(
                64,
                activation=tf.nn.relu,
                kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
                bias_initializer='zeros'
            )
        ]
    )
    return fcn
