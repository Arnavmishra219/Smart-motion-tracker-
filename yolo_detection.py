
from ultralytics import YOLO
import cv2

# Load a lightweight YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("YOLO Human Detection Started")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # Detect only people (class 0)
    results = model(frame, classes=[0], conf=0.65, verbose=False)
    # Draw detection boxes and labels
    annotated_frame = results[0].plot()

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

    cv2.imshow("SmartMotionTracker - YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
