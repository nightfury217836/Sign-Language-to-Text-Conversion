import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template, Response
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")

# ------------------- CONFIG -------------------
DATA_DIR = "data/"               # Dataset folder
IMG_SIZE = 224                   # Updated image size
MODEL_PATH = "modelnet_model.h5" # Path to model
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"}

LABELS = sorted(os.listdir(DATA_DIR))  # Load class labels

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- MEDIAPIPE HANDS -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------- TRAIN MODEL -------------------
def build_and_train_model():
    global LABELS
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode="categorical", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode="categorical", subset="validation"
    )

    LABELS = list(train_gen.class_indices.keys())
    print("Detected classes:", LABELS)

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE,3)),
        MaxPooling2D(2,2),
        Conv2D(64,(3,3),activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128,(3,3),activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(LABELS), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, epochs=20, validation_data=val_gen)
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved to {MODEL_PATH}")
    return model

# ------------------- LOAD OR TRAIN -------------------
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Loaded pre-trained ModelNet model")
else:
    print("⚡ Model not found, starting training...")
    model = build_and_train_model()

# ------------------- HELPERS -------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_crop_hand(frame):
    """Detect hand and crop using MediaPipe"""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None

    h, w, _ = frame.shape
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_coords)) - 20, int(max(x_coords)) + 20
        y_min, y_max = int(min(y_coords)) - 20, int(max(y_coords)) + 20
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        cropped = frame[y_min:y_max, x_min:x_max]
        return cropped
    return None

def preprocess_frame(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_frame(frame):
    cropped = detect_and_crop_hand(frame)
    if cropped is None:
        return "No Hand Detected", 0.0
    processed = preprocess_frame(cropped)
    preds = model.predict(processed, verbose=0)
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))
    return LABELS[class_index], confidence

def extract_frames_and_predict(video_path, step=5):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            label, _ = predict_frame(frame)
            if label != "No Hand Detected":
                sequence.append(label)
        frame_count += 1
    cap.release()
    collapsed = []
    for char in sequence:
        if not collapsed or char != collapsed[-1]:
            collapsed.append(char)
    return " ".join(collapsed)

# ------------------- FLASK ROUTES -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    label, conf = predict_frame(img)
    return jsonify({"prediction": label, "confidence": conf})

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)
    sequence = extract_frames_and_predict(video_path, step=5)
    return jsonify({"prediction": sequence})

# ------------------- LIVE WEBCAM FEED -------------------
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        label, conf = predict_frame(frame)
        display_text = f"{label} ({conf:.2f})" if label != "No Hand Detected" else label
        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/predict_live")
def predict_live():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------- MAIN -------------------
if __name__ == "__main__":
    app.run(debug=True)
