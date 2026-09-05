import cv2
import numpy as np
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "gender_age_prediction_model.keras"

IMG_SIZE = (128, 128)

# DroidCam غالبًا بيكون 0 أو 1 أو 2
CAMERA_INDEX = 0

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ============================================================
# FACE DETECTOR
# ============================================================

FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    raise FileNotFoundError(
        "Could not load haarcascade_frontalface_default.xml"
    )

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    print("Try CAMERA_INDEX = 1 or 2")
    exit()

# Optional resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera started!")
print("Press Q to quit.")

# ============================================================
# REAL-TIME LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # ========================================================
    # PROCESS EACH FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # Add small margin around face
        margin = int(0.15 * min(w, h))

        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face_resized = cv2.resize(
            face_rgb,
            IMG_SIZE
        )

        face_normalized = (
            face_resized.astype(np.float32) / 255.0
        )

        face_input = np.expand_dims(
            face_normalized,
            axis=0
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            face_input,
            verbose=0
        )

        gender_probability = float(
            predictions["gender"][0][0]
        )

        age_normalized = float(
            predictions["age"][0][0]
        )

        # Convert age back to years
        age = age_normalized * 100

        # Keep age within realistic range
        age = np.clip(age, 0, 100)

        # ----------------------------------------------------
        # GENDER
        # ----------------------------------------------------

        if gender_probability >= 0.5:

            gender = "Female"

            gender_confidence = (
                gender_probability * 100
            )

        else:

            gender = "Male"

            gender_confidence = (
                (1 - gender_probability) * 100
            )

        # ----------------------------------------------------
        # DRAW FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # LABEL BACKGROUND
        # ----------------------------------------------------

        label_height = 80

        label_y1 = max(
            0,
            y1 - label_height
        )

        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x2, y1),
            (0, 0, 0),
            -1
        )

        # ----------------------------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Gender: {gender}",
            (x1 + 10, label_y1 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Age: {age:.0f} years",
            (x1 + 10, label_y1 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {gender_confidence:.1f}%",
            (x1 + 10, label_y1 + 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    # ========================================================
    # WINDOW TITLE
    # ========================================================

    cv2.putText(
        frame,
        "Gender & Age Prediction",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        "Gender & Age Prediction",
        frame
    )

    # Q = Quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Camera stopped.")