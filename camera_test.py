
import cv2

print("Python is working!")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened!")
    exit()

print("Camera started successfully!")

while True:
    success, frame = camera.read()

    if not success:
        print("Unable to capture video")
        break

    cv2.imshow("My First Motion Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
print("Camera stopped!")
