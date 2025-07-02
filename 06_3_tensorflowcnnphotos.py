# Importing the libraries
import numpy as np
import tensorflow as tf
import os

# Prepare the training set
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('./datasets/training_set', target_size=(64,64), batch_size=32, class_mode='binary')

# Prepare the test set
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
test_set = test_datagen.flow_from_directory('./datasets/test_set', target_size=(64,64), batch_size=32, class_mode='binary')

# Prepare the neural network
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64,64,3]))    # 1st convolutional layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))    # 2nd convolutional layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))                    # Fully connected layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))                   # Output layer
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the neural network
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Save trained network
output_dir = "./models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
cnn.save(output_dir + "/cat_or_dog.h5")

# Test the neural network
test_image = tf.keras.utils.load_img('./datasets/cat_or_dog.jpg', target_size=(64,64))
test_batch = np.expand_dims(tf.keras.utils.img_to_array(test_image), axis=0)
test_result = cnn.predict(test_batch)
print(test_result[0][0])
print('dog' if test_result[0][0] >= 0.5 else 'cat')