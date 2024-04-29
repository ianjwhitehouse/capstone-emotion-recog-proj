import tensorflow as tf
from keras import regularizers

# happiness model
def happiness_predictor(X_train, n_output):
    input = tf.keras.Input(shape=(X_train.shape[1],))
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer="l2")(input)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer="l2")(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer="l2")(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer="l2")(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer="l2")(x)
    x = tf.keras.layers.Dense(n_output)(x)
    x = tf.keras.activations.softmax(x)

    model = tf.keras.models.Model(input, x)

    return model

# sadness model
def sadness_predictor(X_train, n_output):
    input = tf.keras.Input(shape=(X_train.shape[1],))
    regularizer = regularizers.L1L2(l1=1e-5, l2=1e-5)
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizer)(input)
    x = tf.keras.layers.Dense(384, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(384, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(n_output)(x)
    x = tf.keras.activations.softmax(x)

    model = tf.keras.models.Model(input, x)

    return model

# anger model
def anger_predictor(X_train, n_output):
    input = tf.keras.Input(shape=(X_train.shape[1],))

    regularizer = regularizers.L1L2(l1=1e-4, l2=1e-5)
    
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizer)(input)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(384, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizer)(x)
    x = tf.keras.layers.Dense(n_output)(x)
    x = tf.keras.activations.softmax(x)

    model = tf.keras.models.Model(input, x)

    return model
