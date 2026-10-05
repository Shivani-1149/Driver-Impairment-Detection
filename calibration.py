
import cv2
import dlib
import time
import json
import math
from pathlib import Path

# Load the 68-point facial landmark model
model_path = Path(__file__).with_name(
    "shape_predictor_68_face_landmarks.dat"
)

if not model_path.exists():
    print("Error: Facial landmark model not found!")
    raise SystemExit

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(model_path))

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(points):
    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    raise SystemExit

print("Camera initialized successfully!")
print("Look directly at the camera and keep your eyes open.")
print("Calibration will run for 5 seconds.")

duration = 5
start_time = time.time()
total_frames = 0
face_frames = 0
ear_samples = []

try:
    while time.time() - start_time < duration:
        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        total_frames += 1

        for face in faces:
            face_frames += 1
            landmarks = predictor(gray, face)

            # Draw all 68 facial landmarks
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Get the six points around each eye
            left_eye = [
                (landmarks.part(i).x, landmarks.part(i).y)
                for i in range(36, 42)
            ]

            right_eye = [
                (landmarks.part(i).x, landmarks.part(i).y)
                for i in range(42, 48)
            ]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            average_ear = (left_ear + right_ear) / 2

            ear_samples.append(average_ear)

            cv2.rectangle(
                frame,
                (face.left(), face.top()),
                (face.right(), face.bottom()),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"EAR: {average_ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        remaining = max(0, duration - (time.time() - start_time))

        cv2.putText(
            frame,
            f"Calibration: {remaining:.1f} sec",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Driver Calibration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# Save calibration results
baseline_ear = (
    sum(ear_samples) / len(ear_samples)
    if ear_samples else None
)

baseline = {
    "calibration_duration_seconds": duration,
    "total_frames": total_frames,
    "face_detected_frames": face_frames,
    "face_detection_ratio": (
        face_frames / total_frames if total_frames else 0
    ),
    "baseline_ear": baseline_ear,
    "ear_samples_count": len(ear_samples)
}

with open(Path(__file__).with_name("baseline.json"), "w") as file:
    json.dump(baseline, file, indent=4)

print("\nCalibration completed!")
print("Baseline results saved to baseline.json")
print("Baseline EAR:", baseline_ear)

if baseline_ear is None:
    print("No eye measurements collected. Check lighting and face visibility.")