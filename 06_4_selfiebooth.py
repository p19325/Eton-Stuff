import cv2
import os
from datetime import datetime

output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

window_name = 'Selfie Booth'
cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

print("Press the space bar to take a photo. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        continue
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Selfie Booth', (480, 150), font, 5, (255, 255, 255), 5, cv2.LINE_AA) 

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)

    if key == ord(' '):  
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"selfie_{timestamp}.jpg")
        
        cv2.imwrite(filename, frame)
        print(f"Photo saved as {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()