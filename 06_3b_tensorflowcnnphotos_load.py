# Importing the libraries
import numpy as np
import tensorflow as tf

# Save trained network
cnn = tf.keras.models.load_model("./models/cat_or_dog.h5")

# Test the neural network
test_image = tf.keras.utils.load_img('./datasets/cat_or_dog.jpg', target_size=(64,64))
test_batch = np.expand_dims(tf.keras.utils.img_to_array(test_image), axis=0)
test_result = cnn.predict(test_batch)
print(test_result[0][0])
print('dog' if test_result[0][0] >= 0.5 else 'cat')