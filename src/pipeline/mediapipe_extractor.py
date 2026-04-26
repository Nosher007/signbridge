"""
MediaPipe hand landmark extractor for SignBridge.
Handles both single images (ASL alphabet) and video files (WLASL words).
Output: 63 features per frame (21 landmarks × 3 coords)
"""

import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from google.cloud import storage

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

BUCKET = "signbridge-data"
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 63  # 21 landmarks × 3 (x, y, z)


def extract_landmarks_from_frame(frame, hands_model):
    """
    Extract 63 landmark features from a single BGR frame.
    Returns np.array of shape (63,) or None if no hand detected.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_model.process(rgb)
    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
        return coords.flatten()  # (63,)
    return None


def extract_from_image(image_path, hands_model):
    """
    Extract landmarks from a single image file.
    Returns np.array of shape (63,) — zero-filled if no hand detected.
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    result = extract_landmarks_from_frame(frame, hands_model)
    if result is None:
        return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    return result


def extract_from_image_bytes(img_bytes, hands_model):
    """
    Extract landmarks from image bytes (from GCS blob).
    Returns np.array of shape (63,)
    """
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    result = extract_landmarks_from_frame(frame, hands_model)
    if result is None:
        return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    return result


def extract_from_video_bytes(video_bytes, hands_model, sequence_length=SEQUENCE_LENGTH):
    """
    Extract landmarks from video bytes (from GCS blob).
    Samples/pads to fixed sequence_length.
    Returns np.array of shape (sequence_length, 63).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            lm = extract_landmarks_from_frame(frame, hands_model)
            frames.append(lm if lm is not None else np.zeros(FEATURES_PER_FRAME, dtype=np.float32))
        cap.release()
    finally:
        os.unlink(tmp_path)

    if len(frames) == 0:
        return np.zeros((sequence_length, FEATURES_PER_FRAME), dtype=np.float32)

    frames = np.array(frames, dtype=np.float32)

    # Pad or truncate to sequence_length
    if len(frames) >= sequence_length:
        # Evenly sample sequence_length frames
        indices = np.linspace(0, len(frames) - 1, sequence_length, dtype=int)
        frames = frames[indices]
    else:
        # Pad with zeros at the end
        pad = np.zeros((sequence_length - len(frames), FEATURES_PER_FRAME), dtype=np.float32)
        frames = np.vstack([frames, pad])

    assert frames.shape == (sequence_length, FEATURES_PER_FRAME), \
        f"Expected ({sequence_length}, {FEATURES_PER_FRAME}), got {frames.shape}"
    return frames


def get_hands_model(min_detection_confidence=0.7, min_tracking_confidence=0.5):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    client = storage.Client(project="signbridge-prod")
    bucket = client.bucket(BUCKET)

    with get_hands_model() as hands:
        # Test on 1 ASL image
        print("Testing on ASL image...")
        blob = bucket.blob("raw/asl_alphabet/test/asl_alphabet_test/A_test.jpg")
        img_bytes = blob.download_as_bytes()
        lm = extract_from_image_bytes(img_bytes, hands)
        print(f"  Image landmark shape: {lm.shape}")
        assert lm.shape == (63,), f"Expected (63,), got {lm.shape}"
        print("  PASSED ✓")

        # Test on 1 WLASL video
        print("Testing on WLASL video...")
        blob = bucket.blob("raw/wlasl/wlasl_data/videos/00335.mp4")
        vid_bytes = blob.download_as_bytes()
        seq = extract_from_video_bytes(vid_bytes, hands)
        print(f"  Video sequence shape: {seq.shape}")
        assert seq.shape == (30, 63), f"Expected (30, 63), got {seq.shape}"
        print("  PASSED ✓")

    print("\nAll mediapipe_extractor tests passed.")
