import numpy as np
import pandas as pd
import tensorflow as tf
import os

dataset = pd.read_csv('./datasets/mnist.csv') 
X = dataset.iloc[:, 1:].to_numpy()
y = dataset.iloc[:, 0].to_numpy()
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)

train_sample_size = 2500 
test_sample_size = 1000

X_train = X[:train_sample_size]
y_train = y[:train_sample_size]
X_test = X[train_sample_size:train_sample_size+test_sample_size]
y_test = y[train_sample_size:train_sample_size+test_sample_size]

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

output_dir = './models'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cnn.save(os.path.join(output_dir, 'cnn_multi_layer_tf.h5'))

_, accuracy = cnn.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {accuracy * 100:.2f}%')
