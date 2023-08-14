import tensorflow as tf

keraslayers = tf.keras.layers


def create_encoder(encoder_inputs):
    """
    Create encoder model for membership inference attack.
    Individual attack input components are concatenated and passed to encoder.
    """
    appended = keraslayers.concatenate(encoder_inputs, axis=1)

    encoder = keraslayers.Dense(
        256,
        input_shape=(int(appended.shape[1]),),
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
        bias_initializer='zeros')(appended)
    encoder = keraslayers.Dense(
        128,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
        bias_initializer='zeros')(encoder)
    encoder = keraslayers.Dense(
        64,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
        bias_initializer='zeros')(encoder)
    encoder = keraslayers.Dense(
        1,
        activation=tf.nn.sigmoid,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01),
        bias_initializer='zeros')(encoder)
    return encoder
