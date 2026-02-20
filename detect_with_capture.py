import cv2
import torch
import time
import os
from collections import deque

# ---------------- SETTINGS ----------------
WEIGHTS_PATH = "runs/train/exp11/weights/best.pt"
VIDEO_SOURCE = "road.mp4"  # change if needed
CONF_THRESHOLD = 0.5
PRE_FRAMES = 90   # 3 sec before (30fps)
POST_FRAMES = 90  # 3 sec after
SAVE_FOLDER = "captured_clips"
# ------------------------------------------
COOLDOWN_SECONDS = 5
last_capture_time = 0

REQUIRED_CONSECUTIVE = 3
consecutive_detections = 0

os.makedirs(SAVE_FOLDER, exist_ok=True)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=WEIGHTS_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)

frame_buffer = deque(maxlen=PRE_FRAMES)
capturing = False
capture_frames = []
post_count = 0

print("Starting detection with clip capture...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append(frame.copy())

    # Run detection
    results = model(frame)
    detections = results.xyxy[0]

    anomaly_detected = False

    for *box, conf, cls in detections:
        if conf > CONF_THRESHOLD:
            anomaly_detected = True
           
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)
    
    if anomaly_detected:
        consecutive_detections += 1
    else:
        consecutive_detections = 0 


    current_time = time.time()
    
    if consecutive_detections >= REQUIRED_CONSECUTIVE and not capturing:
        if current_time - last_capture_time > COOLDOWN_SECONDS:
            print("Anomaly detected! Capturing clip...")
            capturing = True
            capture_frames = list(frame_buffer)
            post_count = 0


    if capturing:
        capture_frames.append(frame.copy())
        post_count += 1

        if post_count >= POST_FRAMES:
            timestamp = int(time.time())
            filename = os.path.join(SAVE_FOLDER, f"anomaly_{timestamp}.mp4")

            height, width, _ = capture_frames[0].shape
            out = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (width, height)
            )

            for f in capture_frames:
                out.write(f)

            out.release()
            print(f"Clip saved: {filename}")
            last_capture_time = time.time()
            capturing = False


    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
