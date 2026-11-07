"""FarmFresh: a touchless recipe assistant for MediaPipe Tasks and OpenCV."""

from __future__ import annotations

import math
import os
import time
import urllib.request
from pathlib import Path
from textwrap import wrap

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

RECIPE_NAME = "Garden Tomato Pasta"
STEPS = [
    "Wash the tomatoes and fresh basil under clean running water.",
    "Chop the tomatoes, garlic, and basil on a clean cutting board.",
    "Bring a pot of salted water to a boil and add the pasta.",
    "Warm olive oil in a pan; cook the garlic until fragrant.",
    "Add tomatoes and simmer until saucy, then stir in the basil.",
    "Drain the pasta, toss it with the sauce, and serve warm.",
]

# MediaPipe hand landmarks: wrist 0; thumb 1-4; fingers 5-20.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
FINGER_JOINTS = (
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)
# Relative geometry thresholds; dimensions are measured against palm length.
PINCH_MAX_HAND_RATIO = 0.34
PINCH_MIN_INDEX_ANGLE = 110.0
EXTENDED_MIN_PIP_ANGLE = 150.0
EXTENDED_MIN_REACH = 0.06
CURLED_MAX_PIP_ANGLE = 140.0
CURLED_MAX_REACH = 0.10
THUMB_EXTENDED_MIN_ANGLE = 145.0
OPEN_PALM_MIN_THUMB_SPREAD = 0.45
THUMBS_UP_MIN_RISE = 0.25
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def get_model() -> Path:
    """Download the official task model once when it is not beside this script."""
    if not MODEL_PATH.exists():
        print("Downloading the official MediaPipe hand landmark model...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as exc:
            MODEL_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download hand_landmarker.task. Check your internet "
                f"connection and retry. Source: {MODEL_URL}"
            ) from exc
    return MODEL_PATH


def classify_gesture(hand: list, image_size: tuple[int, int] | None = None) -> tuple[str, float]:
    """Return a gesture and geometry match score (0-100), normalized to palm size."""
    image_width, image_height = image_size or (1, 1)

    def delta(a: int, b: int) -> tuple[float, float]:
        return ((hand[a].x - hand[b].x) * image_width,
                (hand[a].y - hand[b].y) * image_height)

    def distance(a: int, b: int) -> float:
        dx, dy = delta(a, b)
        return math.hypot(dx, dy)

    def angle(a: int, b: int, c: int) -> float:
        """Angle ABC in degrees; 180 degrees means a straight finger."""
        ax, ay = delta(a, b)
        cx, cy = delta(c, b)
        denom = math.hypot(ax, ay) * math.hypot(cx, cy)
        if denom < 1e-8:
            return 0.0
        cosine = max(-1.0, min(1.0, (ax * cx + ay * cy) / denom))
        return math.degrees(math.acos(cosine))

    def clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    # Wrist-to-middle-MCP palm length compensates for hand size and camera distance.
    hand_scale = distance(WRIST, MIDDLE_MCP)
    if hand_scale < 1e-6:
        return "No gesture", 0.0

    pinch_ratio = distance(THUMB_TIP, INDEX_TIP) / hand_scale
    pinch_score = clamp((0.48 - pinch_ratio) / 0.20)
    index_angle = angle(INDEX_MCP, INDEX_PIP, INDEX_DIP)
    index_reach = (distance(WRIST, INDEX_TIP) - distance(WRIST, INDEX_PIP)) / hand_scale
    # A tightly curled index is likely part of a fist; reject that as pinch.
    if (pinch_ratio <= PINCH_MAX_HAND_RATIO
            and (index_angle >= PINCH_MIN_INDEX_ANGLE or index_reach >= 0.02)):
        return "PINCH", max(60.0, pinch_score * 100.0)

    extension_scores, curl_scores = [], []
    extended, curled = [], []
    for mcp, pip, dip, tip in FINGER_JOINTS:
        pip_angle = angle(mcp, pip, dip)
        reach = (distance(WRIST, tip) - distance(WRIST, pip)) / hand_scale
        extension_scores.append(
            clamp((pip_angle - 125.0) / 45.0) * clamp((reach + 0.02) / 0.18)
        )
        curl_scores.append(
            clamp((145.0 - pip_angle) / 55.0) * clamp((0.16 - reach) / 0.22)
        )
        extended.append(
            pip_angle >= EXTENDED_MIN_PIP_ANGLE and reach >= EXTENDED_MIN_REACH
        )
        curled.append(pip_angle <= CURLED_MAX_PIP_ANGLE and reach <= CURLED_MAX_REACH)

    thumb_angle = angle(THUMB_MCP, THUMB_IP, THUMB_TIP)
    thumb_straight_score = clamp((thumb_angle - 125.0) / 50.0)
    thumb_spread = distance(THUMB_TIP, INDEX_MCP) / hand_scale
    thumb_vertical = (
        (hand[THUMB_MCP].y - hand[THUMB_TIP].y) * image_height / hand_scale
    )

    open_score = (
        sum(extension_scores) / 4.0 * 0.75
        + thumb_straight_score * 0.10
        + clamp((thumb_spread - 0.25) / 0.50) * 0.15
    )
    fist_score = sum(curl_scores) / 4.0
    thumbs_score = (
        fist_score * 0.65
        + thumb_straight_score * 0.20
        + clamp((thumb_vertical - 0.12) / 0.45) * 0.15
    )

    # Vertical thumb uses y only; open-palm spread uses a distance, not x direction.
    if (all(curled) and thumb_angle >= THUMB_EXTENDED_MIN_ANGLE
            and thumb_vertical >= THUMBS_UP_MIN_RISE):
        return "THUMBS UP", max(60.0, thumbs_score * 100.0)
    if (all(extended) and thumb_angle >= 140.0
            and thumb_spread >= OPEN_PALM_MIN_THUMB_SPREAD):
        return "OPEN PALM", max(60.0, open_score * 100.0)
    if all(curled):
        return "FIST", max(60.0, fist_score * 100.0)

    return "No gesture", max(pinch_score, open_score, fist_score, thumbs_score) * 100.0


class GestureDebouncer:
    """Require a stable pose, neutral release, and a short action cooldown."""

    def __init__(self, confirm_frames: int = 5, release_frames: int = 5,
                 cooldown_seconds: float = 0.65):
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames
        self.cooldown_seconds = cooldown_seconds
        self.candidate = "No gesture"
        self.candidate_frames = 0
        self.neutral_frames = 0
        self.armed = True
        self.last_action = float("-inf")

    def update(self, gesture: str, now: float) -> bool:
        """Return True once per held gesture after frame and cooldown checks."""
        if gesture == "No gesture":
            self.candidate = "No gesture"
            self.candidate_frames = 0
            self.neutral_frames += 1
            if self.neutral_frames >= self.release_frames:
                self.armed = True
            return False

        self.neutral_frames = 0
        if gesture == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = gesture
            self.candidate_frames = 1

        if (self.armed and self.candidate_frames >= self.confirm_frames
                and now - self.last_action >= self.cooldown_seconds):
            self.armed = False
            self.last_action = now
            return True
        return False


def draw_hand(frame, hand) -> None:
    height, width = frame.shape[:2]
    points = [(int(p.x * width), int(p.y * height)) for p in hand]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (80, 220, 120), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (30, 80, 255), -1)


def draw_text(frame, text: str, position: tuple[int, int], *, scale=0.65,
              color=(245, 245, 245), thickness=2) -> None:
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_panel(frame, gesture: str, confidence: float | None, step_index: int,
               paused: bool, confirmed: bool) -> None:
    height, width = frame.shape[:2]
    panel_height = 228
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, panel_height), (24, 34, 28), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    draw_text(frame, "FarmFresh", (18, 31),
              scale=0.72, color=(150, 240, 170))
    draw_text(frame, "TOUCHLESS KITCHEN ASSISTANT", (150, 29),
              scale=0.43, color=(190, 205, 194), thickness=1)
    draw_text(frame, RECIPE_NAME, (18, 65), scale=0.65)
    draw_text(frame, f"STEP {step_index + 1} / {len(STEPS)}", (18, 96),
              color=(190, 220, 255))
    lines = wrap(STEPS[step_index], width=max(28, (width - 36) // 12))
    for row, line in enumerate(lines[:2]):
        draw_text(frame, line, (18, 124 + row * 25), scale=0.58)
    draw_text(frame, "GESTURES  Open palm: next   Pinch: previous   Fist: pause   Thumbs up: confirm",
              (18, 178), scale=0.39, color=(190, 205, 194), thickness=1)
    status = "PAUSED" if paused else ("STEP CONFIRMED" if confirmed else "READY")
    draw_text(frame, f"STATUS  {status}", (18, 211), scale=0.52,
              color=(80, 220, 255) if paused else (150, 240, 170))
    # A solid footer keeps controls readable over any webcam background.
    cv2.rectangle(frame, (0, height - 68), (width, height), (24, 34, 28), -1)
    draw_text(frame, f"GESTURE  {gesture}", (18, height - 43), scale=0.58,
              color=(255, 220, 100))
    if confidence is None:
        match_text = "STATUS: SHOW HAND"
    elif gesture == "No gesture":
        match_text = "STATUS: ADJUST POSE"
    else:
        match_text = f"MATCH {confidence:.0f}%"
    draw_text(frame, match_text, (width - 175, height - 43), scale=0.48,
              color=(150, 240, 170) if confidence is not None and gesture != "No gesture"
              else (190, 205, 194),
              thickness=1)
    draw_text(frame, "N next  |  P previous  |  Space pause  |  C confirm  |  Q / Esc quit",
              (18, height - 17), scale=0.43, thickness=1)


def run() -> None:
    model_path = get_model()
    mp_image = mp.Image
    running_mode = mp.tasks.vision.RunningMode.VIDEO
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    camera = cv2.VideoCapture(0, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Could not open webcam. Check its connection or camera permissions.")

    step_index = 0
    paused = False
    confirmed = False
    gesture_gate = GestureDebouncer()
    last_timestamp = -1

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Webcam frame could not be read; stopping.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(int(time.monotonic() * 1000), last_timestamp + 1)
                last_timestamp = timestamp
                result = landmarker.detect_for_video(
                    mp_image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                )

                detected = "No gesture"
                confidence = None
                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    draw_hand(frame, hand)
                    detected, confidence = classify_gesture(
                        hand, (frame.shape[1], frame.shape[0])
                    )

                if gesture_gate.update(detected, time.monotonic()):
                    if detected == "OPEN PALM" and not paused:
                        step_index = min(step_index + 1, len(STEPS) - 1)
                        confirmed = False
                    elif detected == "PINCH" and not paused:
                        step_index = max(step_index - 1, 0)
                        confirmed = False
                    elif detected == "FIST":
                        paused = not paused
                    elif detected == "THUMBS UP" and not paused:
                        confirmed = True

                visible_gesture = detected if result.hand_landmarks else "No hand detected"
                draw_panel(frame, visible_gesture, confidence, step_index, paused, confirmed)
                cv2.imshow("FarmFresh - press Q or Esc to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("n") and not paused:
                    step_index = min(step_index + 1, len(STEPS) - 1)
                    confirmed = False
                elif key == ord("p") and not paused:
                    step_index = max(step_index - 1, 0)
                    confirmed = False
                elif key == ord("c") and not paused:
                    confirmed = True
                elif key == ord(" "):
                    paused = not paused
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except (RuntimeError, OSError) as error:
        print(f"FarmFresh: {error}")
