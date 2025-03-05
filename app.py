import os
import cv2
import numpy as np
import base64
import json
from flask import Flask, render_template, request, jsonify, session

app = Flask(__name__)
app.secret_key = "face-recognition-secret"

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
RECOGNIZER_PATH = "recognizer.yml"
LABELS_PATH = "labels.json"

CONFIDENCE_THRESHOLD = 80


def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH) as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f)


def decode_image(b64_data):
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    return face_roi, faces[0]


def retrain_recognizer():
    labels_map = load_labels()
    if not labels_map:
        return False
    faces, labels = [], []
    for label_id, username in labels_map.items():
        user_dir = os.path.join(KNOWN_FACES_DIR, username)
        if not os.path.exists(user_dir):
            continue
        for fname in os.listdir(user_dir):
            if fname.endswith(".jpg"):
                img_path = os.path.join(user_dir, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    faces.append(img)
                    labels.append(int(label_id))
    if not faces:
        return False
    recognizer.train(faces, np.array(labels))
    recognizer.save(RECOGNIZER_PATH)
    return True


# Load recognizer if it exists
if os.path.exists(RECOGNIZER_PATH):
    recognizer.read(RECOGNIZER_PATH)


@app.route("/")
def index():
    labels_map = load_labels()
    users = list(labels_map.values())
    logged_in = session.get("user")
    return render_template("index.html", users=users, logged_in=logged_in)


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    image_data = data.get("image", "")

    if not username:
        return jsonify({"error": "Please enter a username."}), 400
    if not image_data:
        return jsonify({"error": "No image captured."}), 400

    img = decode_image(image_data)
    face_roi, bbox = detect_face(img)
    if face_roi is None:
        return jsonify({"error": "No face detected. Please look at the camera and try again."}), 400

    # Assign label ID
    labels_map = load_labels()
    # Check if username already exists
    existing_id = None
    for lid, name in labels_map.items():
        if name.lower() == username.lower():
            existing_id = int(lid)
            break

    if existing_id is None:
        label_id = len(labels_map)
        labels_map[str(label_id)] = username
        save_labels(labels_map)
    else:
        label_id = existing_id

    # Save face image
    user_dir = os.path.join(KNOWN_FACES_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    count = len([f for f in os.listdir(user_dir) if f.endswith(".jpg")])
    cv2.imwrite(os.path.join(user_dir, f"face_{count}.jpg"), face_roi)

    retrain_recognizer()
    return jsonify({"success": True, "message": f"Face registered for {username}!"})


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    image_data = data.get("image", "")
    if not image_data:
        return jsonify({"error": "No image captured."}), 400

    if not os.path.exists(RECOGNIZER_PATH):
        return jsonify({"error": "No registered users yet. Please register first."}), 400

    img = decode_image(image_data)
    face_roi, bbox = detect_face(img)
    if face_roi is None:
        return jsonify({"error": "No face detected. Please look at the camera."}), 400

    label_id, confidence = recognizer.predict(face_roi)
    labels_map = load_labels()

    if confidence < CONFIDENCE_THRESHOLD:
        username = labels_map.get(str(label_id), "Unknown")
        session["user"] = username
        return jsonify({
            "success": True,
            "username": username,
            "confidence": round(float(confidence), 1),
            "message": f"Welcome back, {username}!"
        })
    else:
        return jsonify({
            "success": False,
            "confidence": round(float(confidence), 1),
            "message": "Face not recognised. Access denied."
        })


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"success": True})


@app.route("/users")
def users():
    labels_map = load_labels()
    return jsonify(list(labels_map.values()))


if __name__ == "__main__":
    app.run(debug=False, port=5013)
