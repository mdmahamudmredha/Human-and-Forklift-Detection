import cv2
import torch
from pathlib import Path
import time
import os

# ✅ Load custom YOLOv5 model from local best.pt file
#model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', source='local')

# ✅ Open webcam (0 for MacBook's default camera)
cap = cv2.VideoCapture(0)

# ✅ Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = None
recording = False

# ✅ Create folder to save recordings
output_dir = Path("recordings")
output_dir.mkdir(exist_ok=True)

print("✅ Webcam started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # ✅ Inference using YOLOv5 model
    results = model(frame)
    detections = results.pandas().xyxy[0]

    # ✅ Get list of detected class names
    detected_classes = detections['name'].tolist()

    # ✅ Check if both person and forklift are detected
    has_person = 'person' in detected_classes
    has_forklift = 'forklift' in detected_classes

    if has_person and has_forklift:
        if not recording:
            filename = output_dir / f"record_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            output = cv2.VideoWriter(str(filename), fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            recording = True
            print(f"🔴 Started recording: {filename}")
        output.write(frame)
    else:
        if recording:
            recording = False
            output.release()
            output = None
            print("🟢 Stopped recording.")

    # ✅ Display annotated frame
    annotated_frame = results.render()[0]
    cv2.imshow("Live Detection", annotated_frame)

    # ✅ Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
if output:
    output.release()
cv2.destroyAllWindows()
