import numpy as np
import pandas as pd

import tensorflow as tf

dataset = pd.read_csv("./datasets/iris.csv").sample(frac=1)

X = dataset[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
y, _ = dataset["class"].factorize()

split_index = int(len(X) * 0.75)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

nn = tf.keras.models.Sequential()
nn.add(tf.keras.Input(shape=(X_train.shape[1],)))
nn.add(tf.keras.layers.Dense(units=10, activation='relu'))
nn.add(tf.keras.layers.Dense(units=len(np.unique(y)), activation='softmax'))

nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn.fit(X_train, y_train, batch_size=15, epochs=100)

y_prediction = np.argmax(nn.predict(X_test), axis=1)

correct_count = np.sum(y_prediction == y_test)
accuracy = 100 * correct_count / len(y_test)
print(f"Accuracy: {accuracy:.1f}%")
