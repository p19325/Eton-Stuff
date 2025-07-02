import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 224
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCL = 1.2
THICK = 2
TXT_COL = (255, 255, 255)
BG_COL = (0, 0, 0)

preprocess = tf.keras.applications.mobilenet_v3.preprocess_input

model = tf.keras.applications.MobileNetV3Large(weights="imagenet")
decode_preds = tf.keras.applications.mobilenet_v3.decode_predictions

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Error: Could not open camera.")

window_name = "MobileNetV3 Live Classifier"
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

predictions = []

print("Press the space bar to classify the current frame. Press 'q' to quit.")

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
    elif key == ord(' '):
        frame_rs = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(frame_rs, cv2.COLOR_BGR2RGB)
        inp = preprocess(rgb.astype("float32"))
        preds = model.predict(np.expand_dims(inp, 0), verbose=0)
        top5 = decode_preds(preds, top=5)[0]
        predictions = [f"{rank}. {label}: {score*100:.1f}%" for rank, (_, label, score) in enumerate(top5, 1)]
        for line in predictions:
            print(line)

cap.release()
cv2.destroyAllWindows()
