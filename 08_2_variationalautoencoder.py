import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and flatten the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32')  / 255.
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Model configuration
input_dim = 784
bottleneck_size = 2
batch_size = 256
no_epochs = 10

# Encoder
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(512, activation='relu')(input_layer)
encoded = tf.keras.layers.Dense(128, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
encoded = tf.keras.layers.Dense(bottleneck_size, activation='linear')(encoded)
encoder = tf.keras.models.Model(input_layer, encoded)

# Decoder
decoder_input = tf.keras.layers.Input(shape=(bottleneck_size,))
decoded = tf.keras.layers.Dense(64, activation='relu')(decoder_input)
decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(512, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
decoder = tf.keras.models.Model(decoder_input, decoded)

# VAE
encoder_decoder = decoder(encoder(input_layer))
vae = tf.keras.models.Model(input_layer, encoder_decoder)
vae.compile(optimizer='adam', loss='mean_squared_error')

vae.summary()

vae.fit(x_train, x_train, 
        epochs=no_epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

fig, ax = plt.subplots(1, 2)
ax[0].scatter(encoded_imgs[:,0], encoded_imgs[:,1], c=y_test, s=8, cmap='tab10')

def onclick(event):
    global flag

    if event.xdata is None or event.ydata is None:
        return

    ix, iy = event.xdata, event.ydata
    latent_vector = np.array([[ix, iy]])
    
    decoded_img = decoder.predict(latent_vector)
    decoded_img = decoded_img.reshape(28, 28)
    ax[1].imshow(decoded_img, cmap='gray')
    plt.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

plt.show() 


