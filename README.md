# 👤 Face Recognition Login

A web app that replaces passwords with your face. Uses OpenCV's LBPH face recogniser and Haar Cascade detection to authenticate users directly from their webcam — register once, log in with just a look.

## Demo

Open the app → Register your face with your name → Switch to Login → Look at the camera → Access granted or denied in real time.

## Features

- Register your face via live webcam in the browser — no photo upload needed
- Login by looking at the camera — zero passwords
- Supports multiple registered users on the same system
- OpenCV LBPH (Local Binary Pattern Histogram) face recognition
- Haar Cascade real-time face detection
- Session-based login state (stay logged in until you log out)
- Runs 100% locally — no cloud, no API, no data sent anywhere
- Confidence score displayed on every login attempt

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Face Detection | OpenCV Haar Cascade Classifier |
| Face Recognition | OpenCV LBPH Face Recogniser |
| Image Capture | JavaScript getUserMedia API (base64 frames) |
| Web Framework | Flask |
| Session Management | Flask sessions |
| Language | Python |

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/manny2341/face-recognition-login.git
cd face-recognition-login
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the app**
```bash
python3 app.py
```

**4. Open in browser**
```
http://127.0.0.1:5013
```
Allow camera access when the browser prompts.

## How It Works

1. **Register:** JavaScript captures a live webcam frame and sends it as base64 to Flask
2. Flask detects the face using **Haar Cascade** → crops and resizes to 200×200 pixels
3. Face image saved to `known_faces/<username>/` on disk
4. **LBPH recogniser** retrained on all saved faces immediately
5. **Login:** New webcam frame captured → LBPH predicts the closest matching face
6. If confidence score is **below 80** → access granted and session started
7. If confidence is too high (face unknown) → access denied

> Lower LBPH confidence = better match. A score of 0 is a perfect match.

## Project Structure

```
face-recognition-login/
├── app.py               # Flask server, Haar Cascade detection, LBPH recognition
├── known_faces/         # Registered face images per user (auto-created)
├── templates/
│   └── index.html       # Register/Login tabs, live webcam feed, result display
├── static/
│   └── style.css        # Dark theme styling
└── requirements.txt
```

## My Other ML Projects

| Project | Description | Repo |
|---------|-------------|------|
| Speech Emotion Recognition | MLP — detect emotion from voice audio | [speech-emotion-recognition](https://github.com/manny2341/speech-emotion-recognition) |
| Emotion Detection | CNN — real-time webcam emotion recognition | [Emotion-Detection](https://github.com/manny2341/Emotion-Detection) |
| Fake News Detector | NLP — TF-IDF fake vs real news | [fake-news-detector](https://github.com/manny2341/fake-news-detector) |
| Crop Disease Detector | EfficientNetV2 — 15 plant diseases | [crop-disease-detector](https://github.com/manny2341/crop-disease-detector) |

## Author

[@manny2341](https://github.com/manny2341)
