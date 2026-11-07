# SHN-MEDIAPIPE

![Project Cover](assets/cover.jpeg)

# FarmFresh - Touchless Kitchen Assistant

### What's the idea?
FarmFresh is a webcam-based recipe helper for Saturday Hack Night Sprint 01: Beyond Gesture. It uses hand tracking so you can move through a recipe while cooking without touching your keyboard.

### Check it out!
- **Live Website:** N/A (local webcam application)
- **GitHub Repo:** [SHN-MEDIAPIPE](https://github.com/hridyasadanand/SHN-MEDIAPIPE)

### What did you use?
- Python 3.14.2
- MediaPipe 1.2.2 Tasks API (`HandLandmarker` video mode)
- OpenCV 5.0.0.93 and NumPy 2.5.4

### The Humans Behind It
- Add your name and GitHub username here.

### Hardware Corner
- A webcam is required. The app opens camera index 0.

### Show Us What You Built!
- Add screenshots, photos, GIFs, or a demo video link here.

### Cool Things It Does
- Tracks a hand and draws its landmarks in the webcam view.
- Open palm: next recipe step.
- Pinch: previous recipe step.
- Closed fist: pause or resume.
- Thumbs up: confirm the current step.
- Includes a six-step Garden Tomato Pasta recipe and keyboard controls.

## Setup and run

Open PowerShell in the project folder. Activate the existing virtual environment if it is not already active, then install the listed packages and start the app:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python farmfresh.py
```

On first launch, FarmFresh downloads the official `hand_landmarker.task` model into the project folder. An internet connection is needed for this first download. Later launches reuse the model.

## Controls

| Input | Action |
| --- | --- |
| Open palm | Next step |
| Pinch thumb and index finger | Previous step |
| Closed fist | Pause or resume |
| Thumbs up | Confirm the current step |
| `N` | Next step |
| `P` | Previous step |
| `Space` | Pause or resume |
| `C` | Confirm current step |
| `Q` or `Esc` | Close the app |

Gestures need several consecutive frames and a neutral hand pose before the next gesture can trigger. If the webcam does not open, check camera permissions and confirm that another application is not using it.

---

**Built with curiosity at Saturday Hack Night.**
