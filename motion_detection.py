
import cv2
import numpy as np

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found!")
    exit()

# Create background subtractor
background = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=False
)

# Remove small noise
kernel = np.ones((5, 5), np.uint8)

print("Motion Detection Started!")
print("Move your hand in front of the camera.")
print("Press Q to exit.")

while True:

    # Capture frame
    success, frame = camera.read()

    if not success:
        break

    # Resize frame
    frame = cv2.resize(frame, (800, 600))

    # Apply background subtraction
    mask = background.apply(frame)

    # Remove unwanted noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find moving object boundaries
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small movements
        if area < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle around movement
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        motion_count += 1

    # Display number of moving regions
    cv2.putText(
        frame,
        f"Motion Detected: {motion_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # Display windows
    cv2.imshow("Motion Tracker", frame)
    cv2.imshow("Motion Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

print("Motion Detection Stopped!")
