import cv2
import dlib
import math
import time
from pathlib import Path


# ==========================================
# LOAD 68-POINT FACIAL LANDMARK MODEL
# ==========================================

model_path = Path(__file__).with_name(
    "shape_predictor_68_face_landmarks.dat"
)

if not model_path.exists():
    print("Error: Facial landmark model not found!")
    raise SystemExit

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(model_path))


# ==========================================
# EAR FUNCTION
# ==========================================

def calculate_ear(points):

    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])
    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ==========================================
# MAR FUNCTION
# ==========================================

def calculate_mar(points):

    vertical_1 = math.dist(points[2], points[10])
    vertical_2 = math.dist(points[4], points[8])
    horizontal = math.dist(points[0], points[6])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    raise SystemExit


# ==========================================
# THRESHOLDS
# ==========================================

EAR_THRESHOLD = 0.24

CLOSED_EYE_SECONDS = 2.0

# Initial value for testing
MAR_THRESHOLD = 0.60


# ==========================================
# VARIABLES
# ==========================================

eyes_closed_start = None


print("Driver monitoring started.")
print("Press Q to quit.")


# ==========================================
# MAIN PROGRAM
# ==========================================

try:

    while True:

        # ------------------------------
        # Read camera frame
        # ------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Could not read camera frame")
            break


        # ------------------------------
        # Convert to grayscale
        # ------------------------------

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        # ------------------------------
        # Detect faces
        # ------------------------------

        faces = detector(gray)


        for face in faces:

            # ------------------------------
            # Detect 68 facial landmarks
            # ------------------------------

            landmarks = predictor(gray, face)


            # ==========================================
            # EYE LANDMARKS
            # ==========================================

            left_eye = [
                (landmarks.part(i).x, landmarks.part(i).y)
                for i in range(36, 42)
            ]

            right_eye = [
                (landmarks.part(i).x, landmarks.part(i).y)
                for i in range(42, 48)
            ]


            # ------------------------------
            # Calculate EAR
            # ------------------------------

            left_ear = calculate_ear(left_eye)

            right_ear = calculate_ear(right_eye)

            ear = (left_ear + right_ear) / 2.0


            # ==========================================
            # MOUTH LANDMARKS
            # ==========================================

            mouth = [
                (landmarks.part(i).x, landmarks.part(i).y)
                for i in range(48, 68)
            ]


            # ------------------------------
            # Calculate MAR
            # ------------------------------

            mar = calculate_mar(mouth)


            # ==========================================
            # EYE CLOSURE DETECTION
            # ==========================================

            if ear <= EAR_THRESHOLD:

                # Start timer
                if eyes_closed_start is None:

                    eyes_closed_start = time.time()


                # Calculate duration
                closed_duration = (
                    time.time() - eyes_closed_start
                )


                # Check prolonged closure
                if closed_duration >= CLOSED_EYE_SECONDS:

                    eye_status = "WARNING: EYES CLOSED TOO LONG"

                else:

                    eye_status = "EYES CLOSED"


                eye_color = (0, 0, 255)


            else:

                # Eyes opened -> reset timer
                eyes_closed_start = None

                eye_status = "EYES OPEN"

                eye_color = (0, 255, 0)


            # ==========================================
            # YAWNING DETECTION
            # ==========================================

            if mar >= MAR_THRESHOLD:

                yawn_status = "YAWNING"

                yawn_color = (0, 0, 255)

            else:

                yawn_status = "NO YAWN"

                yawn_color = (0, 255, 0)


            # ==========================================
            # DRAW EYE LANDMARKS
            # ==========================================

            for i in list(range(36, 42)) + list(range(42, 48)):

                x = landmarks.part(i).x

                y = landmarks.part(i).y

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (0, 255, 0),
                    -1
                )


            # ==========================================
            # DRAW MOUTH LANDMARKS
            # ==========================================

            for i in range(48, 68):

                x = landmarks.part(i).x

                y = landmarks.part(i).y

                cv2.circle(
                    frame,
                    (x, y),
                    2,
                    (255, 0, 0),
                    -1
                )


            # ==========================================
            # DISPLAY EAR
            # ==========================================

            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                eye_color,
                2
            )


            # ==========================================
            # DISPLAY MAR
            # ==========================================

            cv2.putText(
                frame,
                f"MAR: {mar:.3f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


            # ==========================================
            # DISPLAY EYE STATUS
            # ==========================================

            cv2.putText(
                frame,
                eye_status,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                eye_color,
                2
            )


            # ==========================================
            # DISPLAY YAWN STATUS
            # ==========================================

            cv2.putText(
                frame,
                yawn_status,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                yawn_color,
                2
            )


        # ==========================================
        # SHOW CAMERA
        # ==========================================

        cv2.imshow(
            "Driver Impairment Detection",
            frame
        )


        # ==========================================
        # QUIT WITH Q
        # ==========================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# CLEANUP
# ==========================================

finally:

    cap.release()

    cv2.destroyAllWindows()

    print("Camera closed.")