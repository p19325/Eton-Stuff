import cv2
import numpy as np
import time
import threading
import tensorflow as tf

# Global variables
data = []
labels = []
model = None
classifying = False
num_classes = 10  # Number of classes (0-9)

# Capture function
def capture_image(label):
    global data, labels
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (64, 64))  # Resize to a smaller size for simplicity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        data.append(gray)
        labels.append(label)
        print(f"Captured image for label {label}")

# Train function
def train_model():
    global model, data, labels
    if len(data) > 0:
        X = np.array(data).reshape(-1, 64, 64, 1)
        y = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(X, y, epochs=100, validation_split=0.2)
        print("Model trained")
    else:
        print("No data to train the model")

# Classify function
def classify_image():
    global model
    while classifying:
        success, frame = cap.read()
        if success:
            frame = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_data = gray.reshape(1, 64, 64, 1)
            if model is not None:
                prediction = np.argmax(model.predict(input_data), axis=-1)
                print(f"Predicted class: {prediction[0]}")
            time.sleep(1)

cap = cv2.VideoCapture(0)

window_name = 'AI Selfie Booth'
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

print("Press number keys (0-9) to capture images for respective labels, 't' to train the model, and 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        continue

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Selfie Booth', (480, 150), font, 5, (255, 255, 255), 5, cv2.LINE_AA) 

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('t'):
        print("Training the model...")
        train_model()
        if model is not None:
            thread = threading.Thread(target=classify_image)
            classifying = True
            thread.start()
        else:
            print("Model error!")
            break
    elif key in [ord(str(i)) for i in range(0, 10)]:
        label = int(chr(key))
        capture_image(label)

# Release resources
cap.release()
cv2.destroyAllWindows()
classifying = False