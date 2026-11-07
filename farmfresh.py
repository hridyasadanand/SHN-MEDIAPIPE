"""FarmFresh Smart Kitchen Companion using MediaPipe Tasks and OpenCV."""

from __future__ import annotations

import math
import base64
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path
from textwrap import wrap

import cv2
import mediapipe as mp
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

RECIPES_PATH = Path(__file__).with_name("recipes.json")


def load_recipes() -> list[dict]:
    """Load the built-in sample recipes stored with the application."""
    recipes = json.loads(RECIPES_PATH.read_text(encoding="utf-8"))
    if not isinstance(recipes, list) or not recipes:
        raise ValueError("recipes.json must contain a non-empty recipe list")
    for recipe in recipes:
        if not all(key in recipe for key in ("name", "preparation_minutes", "ingredients", "steps")):
            raise ValueError("Each recipe needs a name, preparation time, ingredients, and steps")
        if not recipe["ingredients"] or not recipe["steps"]:
            raise ValueError(f"Recipe {recipe['name']} must have ingredients and steps")
    return recipes


RECIPES = load_recipes()

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


class StepTimer:
    """Small monotonic-clock timer for the current recipe step."""

    def __init__(self, seconds: int = 0):
        self.set_duration(seconds)

    def set_duration(self, seconds: int) -> None:
        self.duration = max(0, int(seconds))
        self.remaining = float(self.duration) if self.duration else None
        self.started_at = None
        self.has_started = False
        self.finished = False

    def remaining_at(self, now: float) -> float | None:
        if self.remaining is None:
            return None
        if self.started_at is None:
            return self.remaining
        return max(0.0, self.remaining - (now - self.started_at))

    def toggle(self, now: float) -> str:
        if self.remaining is None:
            return "No timer for this step"
        if self.started_at is not None:
            self.remaining = self.remaining_at(now)
            self.started_at = None
            return "Timer paused"
        if self.remaining <= 0:
            return "Timer finished - reset to start again"
        self.started_at = now
        self.has_started = True
        self.finished = False
        return "Timer started"

    def reset(self) -> str:
        self.remaining = float(self.duration) if self.duration else None
        self.started_at = None
        self.has_started = False
        self.finished = False
        return "Timer reset" if self.duration else "No timer for this step"

    def update(self, now: float) -> bool:
        """Return True exactly once when the running timer reaches zero."""
        if self.started_at is not None and self.remaining_at(now) == 0:
            self.remaining = 0.0
            self.started_at = None
            self.finished = True
            return True
        return False


def speak_instruction(instruction: str):
    """Use built-in Windows SAPI through PowerShell; no Python speech package."""
    if os.name != "nt":
        return None
    encoded = base64.b64encode(instruction.encode("utf-16le")).decode("ascii")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$text = [System.Text.Encoding]::Unicode.GetString([Convert]::FromBase64String('{encoded}')); "
        "$speaker.Speak($text); $speaker.Dispose()"
    )
    try:
        return subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-Command", command],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return None


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


def draw_panel(frame, state: dict) -> np.ndarray:
    """Draw the camera view and one of the library, ingredients, or cooking screens."""
    cream, card = (238, 246, 233), (250, 252, 247)
    green, dark_green = (55, 105, 61), (37, 76, 43)
    text, muted, pale_green = (52, 62, 49), (104, 119, 99), (216, 234, 210)
    canvas = np.full((640, 1160, 3), cream, dtype=np.uint8)
    recipe = RECIPES[state["recipe_index"]]
    screen = state["screen"]
    gesture = state["gesture"]

    cv2.rectangle(canvas, (0, 0), (1160, 64), dark_green, -1)
    draw_text(canvas, "FarmFresh", (22, 43), scale=1.0, color=(238, 246, 233))
    draw_text(canvas, "SMART KITCHEN COMPANION", (217, 40),
              scale=0.48, color=(201, 222, 194), thickness=1)
    draw_text(canvas, screen.upper(), (1010, 38), scale=0.5,
              color=(201, 238, 182), thickness=2)

    # Keep the complete annotated camera feed visible.
    camera_view = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
    canvas[90:570, 20:660] = camera_view
    cv2.rectangle(canvas, (19, 89), (661, 571), green, 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (680, 86), (1142, 619), card, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (680, 86), (688, 619), green, -1)

    if screen == "library":
        draw_text(canvas, "Built-in sample recipes", (710, 123),
                  scale=0.64, color=dark_green)
        draw_text(canvas, "Stored locally in recipes.json - no external recipe feed.",
                  (710, 151), scale=0.38, color=muted, thickness=1)
        draw_text(canvas, "SELECT A RECIPE", (710, 190), scale=0.43, color=green)
        for index, item in enumerate(RECIPES):
            top = 207 + index * 94
            fill = pale_green if index == state["recipe_index"] else cream
            cv2.rectangle(canvas, (705, top), (1118, top + 76), fill, -1)
            draw_text(canvas, f"{index + 1}. {item['name']}", (720, top + 29),
                      scale=0.55, color=dark_green)
            draw_text(canvas, f"Preparation: about {item['preparation_minutes']} minutes",
                      (720, top + 55), scale=0.40, color=muted, thickness=1)
        draw_text(canvas, "Press 1-3 to select, then Enter for ingredients.",
                  (710, 512), scale=0.43, color=text, thickness=1)

    elif screen == "ingredients":
        draw_text(canvas, recipe["name"], (710, 122), scale=0.67, color=dark_green)
        draw_text(canvas, f"PREP: ABOUT {recipe['preparation_minutes']} MINUTES",
                  (710, 151), scale=0.4, color=green, thickness=1)
        draw_text(canvas, "INGREDIENT CHECKLIST", (710, 184),
                  scale=0.43, color=green)
        for index, ingredient in enumerate(recipe["ingredients"]):
            top = 201 + index * 43
            if index == state["ingredient_cursor"]:
                cv2.rectangle(canvas, (704, top - 20), (1118, top + 16), cream, -1)
            x, y = 715, top - 4
            cv2.rectangle(canvas, (x, y - 13), (x + 18, y + 5), green, 1)
            if state["prepared"][state["recipe_index"]][index]:
                cv2.line(canvas, (x + 3, y - 4), (x + 8, y + 1), green, 2)
                cv2.line(canvas, (x + 8, y + 1), (x + 16, y - 10), green, 2)
            label = f"{ingredient['quantity']}  {ingredient['name']}"
            draw_text(canvas, label, (745, y + 1), scale=0.48, color=text, thickness=1)
        draw_text(canvas, "Up/Down move   Space checks item", (710, 574),
                  scale=0.4, color=muted, thickness=1)
        draw_text(canvas, "Enter starts cooking   B returns to recipes",
                  (710, 599), scale=0.4, color=muted, thickness=1)

    else:  # cooking screen
        steps = recipe["steps"]
        step = steps[state["step_index"]]
        draw_text(canvas, recipe["name"], (710, 122), scale=0.67, color=dark_green)
        draw_text(canvas,
                  f"STEP {state['step_index'] + 1} OF {len(steps)}  |  PREP {recipe['preparation_minutes']} MIN",
                  (710, 153), scale=0.41, color=green, thickness=1)
        for row, line in enumerate(wrap(step["instruction"], width=39)[:4]):
            draw_text(canvas, line, (710, 205 + row * 31),
                      scale=0.56, color=text, thickness=1)
        progress = (state["step_index"] + 1) / max(1, len(steps))
        draw_text(canvas, "RECIPE PROGRESS", (710, 345),
                  scale=0.4, color=muted, thickness=1)
        cv2.rectangle(canvas, (710, 356), (1118, 370), pale_green, -1)
        cv2.rectangle(canvas, (710, 356), (710 + int(408 * progress), 370), green, -1)

        draw_text(canvas, "STEP TIMER", (710, 405), scale=0.42, color=green)
        if state["timer_remaining"] is None:
            timer_label = "No timer for this step"
        else:
            seconds = max(0, int(state["timer_remaining"] + 0.999))
            timer_label = f"{seconds // 60:02}:{seconds % 60:02}"
        draw_text(canvas, timer_label, (710, 454), scale=1.05, color=dark_green)
        if state["timer_finished"]:
            timer_status = "TIMER FINISHED"
        elif state["timer_active"]:
            timer_status = "RUNNING  |  T or Space pauses"
        elif state["timer_remaining"] is not None:
            timer_status = "PAUSED" if state["timer_started"] else "READY"
        else:
            timer_status = "This step has no timer"
        draw_text(canvas, timer_status, (710, 480), scale=0.4, color=green, thickness=1)
        draw_text(canvas, "T/Space start-pause  |  R reset  |  N/P steps",
                  (710, 523), scale=0.38, color=muted, thickness=1)
        voice_status = "VOICE ON" if state["voice_enabled"] else "VOICE OFF"
        draw_text(canvas, f"{voice_status}  (V toggles, S reads this step)",
                  (710, 551), scale=0.4, color=muted, thickness=1)
        if state["awaiting_release"]:
            gesture_hint = "Release to neutral before another gesture."
        elif gesture == "No hand detected":
            gesture_hint = "Show one hand; keep it inside the camera view."
        elif gesture in ("OPEN PALM", "PINCH"):
            gesture_hint = f"Hold {gesture.lower()} steadily: {state['recognition_frames']}/5"
        else:
            gesture_hint = "Open palm: next step  |  Pinch: previous step"
        draw_text(canvas, gesture_hint, (710, 580), scale=0.37, color=green, thickness=1)
        cv2.rectangle(canvas, (710, 588), (1118, 596), pale_green, -1)
        cv2.rectangle(canvas, (710, 588),
                      (710 + int(408 * max(0.0, min(1.0, state["recognition_progress"]))), 596),
                      green, -1)

    # Compact live gesture status and a helpful camera prompt.
    if gesture == "No hand detected":
        footer = "NO HAND DETECTED  |  Place one hand in the camera view"
    elif state["feedback"]:
        footer = state["feedback"]
    elif screen != "cooking":
        footer = "Open-palm and pinch controls are active on the cooking screen"
    elif gesture in ("OPEN PALM", "PINCH"):
        action = "Next step" if gesture == "OPEN PALM" else "Previous step"
        match = f" | match {state['confidence']:.0f}%" if state["confidence"] is not None else ""
        footer = f"DETECTED: {gesture}  |  ACTION: {action}{match}"
    else:
        footer = "Show an open palm or pinch thumb and index finger"
    footer_bg = pale_green if state["feedback"] else card
    cv2.rectangle(canvas, (17, 582), (663, 621), footer_bg, -1)
    cv2.rectangle(canvas, (17, 582), (663, 621), green, 1, cv2.LINE_AA)
    draw_text(canvas, footer, (22, 612), scale=0.42, color=dark_green, thickness=1)
    draw_text(canvas, "Enter continue   B back   1-3 recipes   V voice   Q/Esc quit",
              (22, 633), scale=0.38, color=muted, thickness=1)
    return canvas


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

    screen = "library"
    recipe_index = step_index = ingredient_cursor = 0
    prepared = [[False] * len(recipe["ingredients"]) for recipe in RECIPES]
    timer = StepTimer(RECIPES[0]["steps"][0].get("timer_seconds", 0))
    voice_enabled = False
    speech_process = None
    gesture_gate = GestureDebouncer()
    last_timestamp = -1
    feedback = ""
    feedback_until = 0.0

    def current_instruction() -> str:
        return RECIPES[recipe_index]["steps"][step_index]["instruction"]

    def speak_current() -> None:
        nonlocal speech_process, feedback, feedback_until
        if speech_process is not None and speech_process.poll() is None:
            speech_process.terminate()
        speech_process = speak_instruction(current_instruction())
        if speech_process is None:
            feedback = "Windows voice is unavailable"
            feedback_until = time.monotonic() + 2.0

    def select_recipe(index: int, stay_in_library: bool) -> None:
        nonlocal recipe_index, step_index, ingredient_cursor, screen, timer, speech_process
        nonlocal feedback, feedback_until
        if speech_process is not None and speech_process.poll() is None:
            speech_process.terminate()
        recipe_index = index
        step_index = 0
        ingredient_cursor = 0
        screen = "library" if stay_in_library else "ingredients"
        timer.set_duration(RECIPES[recipe_index]["steps"][0].get("timer_seconds", 0))
        feedback = f"Selected {RECIPES[recipe_index]['name']}"
        feedback_until = time.monotonic() + 1.5

    def change_step(delta: int, now: float) -> None:
        nonlocal step_index, timer, feedback, feedback_until
        steps = RECIPES[recipe_index]["steps"]
        next_index = max(0, min(step_index + delta, len(steps) - 1))
        if next_index == step_index:
            feedback = "Already at the first step" if delta < 0 else "Recipe complete"
        else:
            step_index = next_index
            timer.set_duration(steps[step_index].get("timer_seconds", 0))
            feedback = "Next cooking step" if delta > 0 else "Previous cooking step"
            if voice_enabled:
                speak_current()
        feedback_until = now + 1.5

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

                visible_gesture = "No hand detected" if not result.hand_landmarks else "No gesture"
                confidence = None
                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    draw_hand(frame, hand)
                    label, score = classify_gesture(hand, (frame.shape[1], frame.shape[0]))
                    if label in ("OPEN PALM", "PINCH"):
                        visible_gesture, confidence = label, score

                now = time.monotonic()
                gesture_input = visible_gesture if screen == "cooking" else "No gesture"
                gesture_triggered = gesture_gate.update(gesture_input, now)
                if gesture_triggered and screen == "cooking":
                    change_step(1 if gesture_input == "OPEN PALM" else -1, now)

                if timer.update(now):
                    feedback = "Cooking timer finished!"
                    feedback_until = now + 5.0
                    if winsound is not None:
                        try:
                            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                        except RuntimeError:
                            pass

                if now >= feedback_until:
                    feedback = ""
                progress = 0.0
                if gesture_triggered:
                    progress = 1.0
                elif gesture_input != "No gesture" and gesture_gate.armed:
                    progress = min(
                        gesture_gate.candidate_frames / gesture_gate.confirm_frames, 1.0
                    )
                timer_remaining = timer.remaining_at(now)
                view_state = {
                    "screen": screen,
                    "recipe_index": recipe_index,
                    "step_index": step_index,
                    "ingredient_cursor": ingredient_cursor,
                    "prepared": prepared,
                    "gesture": visible_gesture,
                    "confidence": confidence,
                    "recognition_progress": progress,
                    "recognition_frames": gesture_gate.candidate_frames,
                    "awaiting_release": screen == "cooking" and not gesture_gate.armed
                    and gesture_input != "No gesture",
                    "timer_remaining": timer_remaining,
                    "timer_active": timer.started_at is not None,
                    "timer_started": timer.has_started,
                    "timer_finished": timer.finished,
                    "voice_enabled": voice_enabled,
                    "feedback": feedback,
                }
                dashboard = draw_panel(frame, view_state)
                cv2.imshow("FarmFresh - Smart Kitchen Companion", dashboard)

                key_code = cv2.waitKeyEx(1)
                key = key_code & 0xFF
                key_char = chr(key).lower() if 32 <= key <= 126 else ""
                enter = key in (10, 13)
                up_key = key_code in (2490368, 65362)
                down_key = key_code in (2621440, 65364)
                if key in (ord("q"), 27):
                    break

                if key_char in ("1", "2", "3"):
                    select_recipe(int(key_char) - 1, stay_in_library=(screen == "library"))
                elif screen == "library":
                    if up_key:
                        recipe_index = (recipe_index - 1) % len(RECIPES)
                    elif down_key:
                        recipe_index = (recipe_index + 1) % len(RECIPES)
                    elif enter:
                        screen = "ingredients"
                        ingredient_cursor = 0
                        feedback = "Mark ingredients as ready, then press Enter"
                        feedback_until = now + 2.0
                elif screen == "ingredients":
                    count = len(RECIPES[recipe_index]["ingredients"])
                    if up_key:
                        ingredient_cursor = (ingredient_cursor - 1) % count
                    elif down_key:
                        ingredient_cursor = (ingredient_cursor + 1) % count
                    elif key == ord(" "):
                        prepared[recipe_index][ingredient_cursor] = not prepared[recipe_index][ingredient_cursor]
                    elif enter:
                        screen = "cooking"
                        step_index = 0
                        timer.set_duration(RECIPES[recipe_index]["steps"][0].get("timer_seconds", 0))
                        if voice_enabled:
                            speak_current()
                elif screen == "cooking":
                    if key_char == "n":
                        change_step(1, now)
                    elif key_char == "p":
                        change_step(-1, now)
                    elif key_char in ("t",) or key == ord(" "):
                        feedback = timer.toggle(now)
                        feedback_until = now + 1.5
                    elif key_char == "r":
                        feedback = timer.reset()
                        feedback_until = now + 1.5
                    elif key_char == "s":
                        speak_current()
                    elif enter:
                        change_step(1, now)

                if key_char == "b":
                    if screen == "cooking":
                        screen = "ingredients"
                    elif screen == "ingredients":
                        screen = "library"
                elif key_char == "v":
                    voice_enabled = not voice_enabled
                    if voice_enabled and screen == "cooking":
                        speak_current()
                    elif not voice_enabled and speech_process is not None:
                        if speech_process.poll() is None:
                            speech_process.terminate()
                        feedback = "Voice instructions off"
                    else:
                        feedback = "Voice instructions on" if voice_enabled else "Voice instructions off"
                    feedback_until = time.monotonic() + 1.5
    finally:
        if speech_process is not None and speech_process.poll() is None:
            speech_process.terminate()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        run()
    except (RuntimeError, OSError) as error:
        print(f"FarmFresh: {error}")
