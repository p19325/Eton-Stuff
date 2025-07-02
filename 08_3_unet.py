import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load the dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# 2. Normalise and add the channel dim that conv-layers expect  ➜  (28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = np.expand_dims(x_train, axis=-1)        # shape (60000, 28, 28, 1)
x_test  = np.expand_dims(x_test,  axis=-1)        # shape (10000, 28, 28, 1)

# 3. Model configuration
input_shape        = (28, 28, 1)
base_filters       = 32           # “width” of the net – double each down-step
dropout_rate       = 0.2
batch_size         = 128
no_epochs          = 5
validation_split   = 0.1
learning_rate      = 1e-3

# 4. U-Net encoder (down-sampling path)  ────────────────────────────────────────
inputs = tf.keras.layers.Input(shape=input_shape)

# ── Level 0
c0 = tf.keras.layers.Conv2D(base_filters, 3, padding="same", activation="relu")(inputs)
c0 = tf.keras.layers.BatchNormalization()(c0)
c0 = tf.keras.layers.Conv2D(base_filters, 3, padding="same", activation="relu")(c0)
p0 = tf.keras.layers.MaxPooling2D()(c0)

# ── Level 1
c1 = tf.keras.layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(p0)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(c1)
p1 = tf.keras.layers.MaxPooling2D()(c1)

# ── Bottleneck
bn = tf.keras.layers.Conv2D(base_filters*4, 3, padding="same", activation="relu")(p1)
bn = tf.keras.layers.Dropout(dropout_rate)(bn)
bn = tf.keras.layers.Conv2D(base_filters*4, 3, padding="same", activation="relu")(bn)

# 5. U-Net decoder (up-sampling path)  ───────────────────────────────────────────
# ── Level 1
u1 = tf.keras.layers.Conv2DTranspose(base_filters*2, 2, strides=2, padding="same")(bn)
cat1 = tf.keras.layers.concatenate([u1, c1])
c2 = tf.keras.layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(cat1)
c2 = tf.keras.layers.BatchNormalization()(c2)
c2 = tf.keras.layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(c2)

# ── Level 0
u0 = tf.keras.layers.Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c2)
cat0 = tf.keras.layers.concatenate([u0, c0])
c3 = tf.keras.layers.Conv2D(base_filters, 3, padding="same", activation="relu")(cat0)
c3 = tf.keras.layers.BatchNormalization()(c3)
c3 = tf.keras.layers.Conv2D(base_filters, 3, padding="same", activation="relu")(c3)

# 6. Output layer – one channel, sigmoid to reconstruct grayscale digit
outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid")(c3)

# 7. Build & compile the model
unet = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="mini_unet_mnist")
unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy")

unet.summary()

# 8. Train
unet.fit(x_train, x_train,             # self-reconstruction target
         epochs=no_epochs,
         batch_size=batch_size,
         shuffle=True,
         validation_split=validation_split)

# 9. Reconstruct some test digits
reconstructed = unet.predict(x_test)

# 10. Visualise
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])

    # reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
plt.show()
