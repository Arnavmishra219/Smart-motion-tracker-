

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("SmartMotionTracker - Person Tracking Started")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect and track only people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4,
        tracker="bytetrack_custom.yaml",
        verbose=False
    )

    
    annotated_frame = results[0].plot()

    # Print tracking information
    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.id is not None:
                track_id = int(box.id[0])
                confidence = float(box.conf[0])

                print(
                    f"ID: {track_id} | "
                    f"Confidence: {confidence:.2f}"
                )

    # Count detected people
    people_count = len(results[0].boxes)

    cv2.putText(
        annotated_frame,
        f"People Count: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("SmartMotionTracker - Person Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
