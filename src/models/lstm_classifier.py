"""
WLASL Word-Level Sign Classifier — three model architectures for comparison.

Model 1: Landmark Sequence LSTM  — (30, 63) landmark features → 100 classes
Model 2: Baseline LSTM from scratch — same architecture, used as lower-bound reference
Model 3: MobileNetV2 + LSTM — (30, 1280) CNN features → 100 classes

Usage:
    from src.models.lstm_classifier import build_landmark_lstm, build_mobilenetv2_lstm
"""

import tensorflow as tf
from tensorflow.keras import layers, models

NUM_CLASSES = 100
SEQ_LEN     = 30
LANDMARK_DIM = 63
MV2_FEATURE_DIM = 1280


def build_landmark_lstm(seq_len=SEQ_LEN, feature_dim=LANDMARK_DIM,
                         num_classes=NUM_CLASSES, dropout=0.3):
    """
    LSTM trained on MediaPipe landmark sequences.
    Input: (batch, 30, 63)
    Fast to train on CPU — used as lightweight baseline.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_len, feature_dim)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(64),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ], name="landmark_lstm")
    return model


def build_mobilenetv2_lstm(seq_len=SEQ_LEN, feature_dim=MV2_FEATURE_DIM,
                            num_classes=NUM_CLASSES, dropout=0.3):
    """
    LSTM trained on MobileNetV2 frame-level feature sequences.
    Input: (batch, 30, 1280) — pre-extracted CNN features
    MobileNetV2 feature extraction is done offline (once) and cached to GCS.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_len, feature_dim)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(64),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax"),
    ], name="mobilenetv2_lstm")
    return model


def extract_mobilenetv2_features(video_frames, base_model):
    """
    Extract MobileNetV2 features from a sequence of video frames.

    Args:
        video_frames: numpy array of shape (30, 224, 224, 3), values 0-255
        base_model: MobileNetV2 with include_top=False, pooling='avg'

    Returns:
        features: numpy array of shape (30, 1280)
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    import numpy as np

    frames = preprocess_input(video_frames.astype("float32"))  # (30, 224, 224, 3)
    features = base_model.predict(frames, verbose=0)           # (30, 1280)
    return features


def build_feature_extractor():
    """MobileNetV2 as a frozen frame-level feature extractor."""
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(224, 224, 3),
    )
    base.trainable = False
    return base
