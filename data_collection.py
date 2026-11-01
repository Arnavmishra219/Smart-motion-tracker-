
import cv2
import numpy as np
import pandas as pd
import math
import time
import os

# Create data folder
os.makedirs("data", exist_ok=True)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

# Background subtraction
background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

kernel = np.ones((5, 5), np.uint8)

# Tracking database
objects = {}
next_id = 1

MAX_DISTANCE = 70

# Store collected data
records = []

start_time = time.time()

print("Data Collection Started!")
print("Move objects in front of camera.")
print("Press Q to stop and save data.")

while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.resize(frame, (800, 600))

    current_time = time.time()
    elapsed = current_time - start_time

    # Detect motion
    mask = background.apply(frame)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    detections = []

    for contour in contours:

        if cv2.contourArea(contour) < 1200:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cx = x + w // 2
        cy = y + h // 2

        detections.append((x, y, w, h, cx, cy))

    updated_objects = {}

    for x, y, w, h, cx, cy in detections:

        best_id = None
        min_distance = MAX_DISTANCE

        for object_id, data in objects.items():

            if object_id in updated_objects:
                continue

            px, py = data["center"]

            distance = math.hypot(cx - px, cy - py)

            if distance < min_distance:
                min_distance = distance
                best_id = object_id

        # New object
        if best_id is None:

            best_id = next_id
            next_id += 1

            distance = 0
            speed = 0
            acceleration = 0
            path = [(cx, cy)]

        else:

            old = objects[best_id]

            px, py = old["center"]

            dt = max(current_time - old["time"], 0.001)

            distance = math.hypot(cx - px, cy - py)

            speed = distance / dt

            acceleration = (
                speed - old["speed"]
            ) / dt

            path = old["path"] + [(cx, cy)]

        path = path[-30:]

        # Update tracking database
        updated_objects[best_id] = {
            "center": (cx, cy),
            "time": current_time,
            "speed": speed,
            "path": path
        }

        # Save observation
        records.append({
            "object_id": best_id,
            "time_seconds": round(elapsed, 3),
            "x": cx,
            "y": cy,
            "distance_pixels": round(distance, 2),
            "speed_pixels_per_second": round(speed, 2),
            "acceleration_pixels_per_second2": round(acceleration, 2)
        })

        # Draw tracking box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"ID {best_id} | {speed:.1f} px/s",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # Draw trajectory
        for i in range(1, len(path)):

            cv2.line(
                frame,
                path[i - 1],
                path[i],
                (255, 100, 0),
                2
            )

    objects = updated_objects

    cv2.putText(
        frame,
        f"Tracked Objects: {len(objects)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Records: {len(records)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Smart Motion Tracker", frame)
    cv2.imshow("Motion Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# Save dataset
if records:

    df = pd.DataFrame(records)

    df.to_csv(
        "data/motion_data.csv",
        index=False
    )

    print("\nDATA COLLECTION COMPLETE!")
    print("Total records:", len(df))
    print("Unique object IDs:", df["object_id"].nunique())
    print("Saved to data/motion_data.csv")

else:
    print("No motion data collected.")
