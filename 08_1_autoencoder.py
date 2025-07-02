import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten the data
x_train = x_train.astype('float64') / 255.0
x_train = x_train.reshape((-1, 784))
x_test = x_test.astype('float64') / 255.0
x_test = x_test.reshape((-1, 784))

# Model configuration
input_dim = 784
encoding_dim = 196
batch_size = 256
no_epochs = 10
validation_split = 0.2

# Encoder
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder_first_dense_layer = tf.keras.layers.Dense(encoding_dim*2, activation='relu')(input_layer)  # First dense layer in the encoder
encoder_batch_norm = tf.keras.layers.BatchNormalization()(encoder_first_dense_layer)  # Batch normalization after first dense layer
encoder_dropout = tf.keras.layers.Dropout(0.2)(encoder_batch_norm)  # Dropout to reduce overfitting
encoder_second_dense_layer = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoder_dropout)  # Second dense layer in the encoder

# Decoder
decoder_first_dense_layer = tf.keras.layers.Dense(encoding_dim*2, activation='relu')(encoder_second_dense_layer)  # First dense layer in the decoder
decoder_batch_norm = tf.keras.layers.BatchNormalization()(decoder_first_dense_layer)  # Batch normalization after first dense layer
decoder_dropout = tf.keras.layers.Dropout(0.2)(decoder_batch_norm)  # Dropout to reduce overfitting
decoder_second_dense_layer = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder_dropout)  # Second dense layer in the decoder

# Autoencoder
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder_second_dense_layer)  # Creating the autoencoder model
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')  # Compiling the model with Adam optimizer

autoencoder.summary()

# Train the autoencoder
autoencoder.fit(x_train, x_train,
                epochs=no_epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_split=validation_split)

# Encode and decode some digits
encoded_imgs = autoencoder.predict(x_test)
decoded_imgs = autoencoder.predict(encoded_imgs)

# Use Matplotlib to visualize the reconstruction
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
