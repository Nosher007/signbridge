"""
ASL Alphabet classifier — three model architectures for comparison.

Model 1: Baseline MLP on MediaPipe landmarks (63 features → 29 classes)
Model 2: Baseline CNN from scratch on raw images (224×224×3 → 29 classes)
Model 3: MobileNetV2 transfer learning on raw images (224×224×3 → 29 classes)

Usage:
    from src.models.cnn_classifier import build_landmark_mlp, build_mobilenetv2
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications

NUM_CLASSES = 29
IMAGE_SIZE  = 224


def build_landmark_mlp(input_dim=63, num_classes=NUM_CLASSES, dropout=0.4):
    """
    Baseline MLP trained on 63 MediaPipe landmark features.
    Fast to train on CPU — used as lower-bound baseline and for real-time inference.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout / 2),
        layers.Dense(num_classes, activation="softmax"),
    ], name="landmark_mlp")
    return model


def build_baseline_cnn(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    """
    Baseline CNN from scratch — 4-layer conv network on raw images.
    Lower-bound reference with no pretrained knowledge.
    """
    model = models.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        # Block 1
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        # Block 2
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        # Block 3
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        # Block 4
        layers.Conv2D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        # Head
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="baseline_cnn")
    return model


def build_mobilenetv2(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES,
                      dropout=0.4, dense_units=256):
    """
    MobileNetV2 transfer learning — hypothesis for best model.
    Phase 1: freeze base, train head only.
    Phase 2: unfreeze top 30 layers, fine-tune end-to-end.
    """
    base = applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False  # Phase 1: frozen

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_asl")
    return model, base


def unfreeze_top_layers(base_model, num_layers=30):
    """Unfreeze the top N layers of the base model for Phase 2 fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfroze top {num_layers} layers ({trainable} trainable layers in base)")
