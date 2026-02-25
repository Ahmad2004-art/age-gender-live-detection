import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
MODEL_PATH = "age_gender_model.keras"  

model = tf.keras.models.load_model(MODEL_PATH)

def gender_label(gid: int) -> str:
    return "Male" if gid == 0 else "Female"


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing VideoCapture index (0/1).")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
       
        pad = int(0.15 * w)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, IMG_SIZE)
        face_rgb = face_rgb.astype(np.float32)
        face_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(face_rgb)
        x_in = np.expand_dims(face_rgb, axis=0)

        pred = model.predict(x_in, verbose=0)
        g_probs = pred["gender"][0]
        age = float(pred["age"][0][0])

        gid = int(np.argmax(g_probs))
        conf = float(np.max(g_probs))

        label = f"{gender_label(gid)} ({conf:.2f}) | Age: {age:.0f}"

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Live Age & Gender Detection - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()