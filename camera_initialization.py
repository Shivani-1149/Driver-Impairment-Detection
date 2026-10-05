import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera initialized successfully!")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set frame rate
cap.set(cv2.CAP_PROP_FPS, 30)

while True:

    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Show the camera frame
    cv2.imshow("Driver Monitoring Camera", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()