import cv2
import numpy as np
import tensorflow as tf
import time
import threading

IMG_SIZE = 224
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCL = 1.2
THICK = 2
TXT_COL = (255, 255, 255)
BG_COL = (0, 0, 0)

NUM_CLASSES = 10

data = []
labels = []
model = None

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Error: Could not open camera.")

window_name = "MobileNetV3 Transfer Learning"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

predictions = []

def capture_image(lbl):
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        data.append(frame)
        labels.append(lbl)
        print(f"Captured frame for label {lbl} | total samples: {len(data)}")

def train_model():
    global model
    if not data:
        print("No data captured!")
        return
    print("Starting training...")
    
    EPOCHS_HEAD = 5
    EPOCHS_FINE = 5
    LEARNING_RATE_HEAD = 3e-3
    LEARNING_RATE_FINE = 1e-4
    BATCH_SIZE = 32
    
    X = np.stack(data).astype("float32")
    y = np.array(labels).astype("int32")

    # Setup the base model
    base = tf.keras.applications.MobileNetV3Small(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False

    # Define our model to be trained, using the base model
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Initial training phase, prepared for our user-defined classes
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_HEAD), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=EPOCHS_HEAD, batch_size=BATCH_SIZE, validation_split=0.2)
    
    # Fine tuning on input data
    base.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE_FINE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=EPOCHS_FINE, batch_size=BATCH_SIZE, validation_split=0.2)
    
    model.summary()
    print("Training complete, starting live classification...")

def classify_stream():
    global classifying
    while classifying:
        success, frame = cap.read()
        if not success:
            continue
        frame_rs = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        input = tf.keras.applications.mobilenet_v3.preprocess_input(frame_rs.astype("float32"))
        pred = np.argmax(model.predict(np.expand_dims(input, 0), verbose=0), axis=-1)[0]
        predictions.clear()
        predictions.append(f"Prediction: {pred}")
        time.sleep(1)

classifying = False

print("Press 0-9 to capture, t to train, q to quit.")

while True:
    success, frame = cap.read()
    if not success:
        continue

    if predictions:
        y0 = 260
        for rank, text in enumerate(predictions, 1):
            y = y0 + rank * 50
            cv2.putText(frame, text, (50, y), FONT, FONT_SCL, BG_COL, THICK + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (50, y), FONT, FONT_SCL, TXT_COL, THICK, cv2.LINE_AA)

    cv2.imshow(window_name, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('t'):
        train_model()
        if model is not None:
            classifying = True
            threading.Thread(target=classify_stream, daemon=True).start()
    elif key in [ord(str(i)) for i in range(NUM_CLASSES)]:
        capture_image(int(chr(key)))

classifying = False
cap.release()
cv2.destroyAllWindows()
