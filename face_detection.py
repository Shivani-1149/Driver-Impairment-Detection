import cv2
import dlib

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize Dlib face detector
detector = dlib.get_frontal_face_detector()

# Check if camera opened
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Camera initialized successfully!")

while True:

    # Read a frame from camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    # Draw rectangle around detected faces
    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    # Display camera
    cv2.imshow("Driver Face Detection", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()

# Close windows
cv2.destroyAllWindows()