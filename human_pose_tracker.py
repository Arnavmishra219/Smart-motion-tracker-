
import cv2
import mediapipe as mp
import pandas as pd
import time
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Select body landmarks
BODY_PARTS = {
    "head": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST
}

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not detected!")
    exit()

records = []
start_time = time.time()

print("Human Pose Tracker Started!")
print("Move your head, shoulders and hands.")
print("Press Q to stop.")

# Start pose detection
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Camera frame not received.")
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Detect pose
        results = pose.process(rgb_frame)

        rgb_frame.flags.writeable = True

        current_time = time.time() - start_time

        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            for part_name, landmark_type in BODY_PARTS.items():

                point = landmarks[landmark_type.value]

                if point.visibility > 0.5:

                    height, width, _ = frame.shape

                    x = int(point.x * width)
                    y = int(point.y * height)

                    # Draw landmark
                    cv2.circle(
                        frame,
                        (x, y),
                        7,
                        (0, 255, 0),
                        -1
                    )

                    # Display body part name
                    cv2.putText(
                        frame,
                        part_name,
                        (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    records.append({
                        "time_seconds": current_time,
                        "body_part": part_name,
                        "x": x,
                        "y": y,
                        "visibility": float(point.visibility)
                    })

            # Draw complete body skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        cv2.imshow("Human Pose Tracker - Phase 5A", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save data
os.makedirs("data", exist_ok=True)

if records:
    df = pd.DataFrame(records)
    df.to_csv("data/human_pose_data.csv", index=False)

    print("\nTracking completed!")
    print("CSV saved: data/human_pose_data.csv")
    print("Total records:", len(df))
else:
    print("No body landmarks detected. Please try again.")
