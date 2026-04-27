"""
Day 3 — ASL Alphabet Training Script

Trains all three models for comparison:
  Model 1: Landmark MLP  (CPU-friendly, runs on GCP VM)
  Model 2: Baseline CNN  (GPU recommended, Kaggle T4)
  Model 3: MobileNetV2  (GPU required, Kaggle T4, 2-phase)

Usage (VM — MLP only):
    python src/models/train_asl.py --model mlp

Usage (Kaggle — all models):
    python src/models/train_asl.py --model cnn
    python src/models/train_asl.py --model mobilenetv2
    python src/models/train_asl.py --model mobilenetv2 --phase 2 --phase1_ckpt gs://signbridge-data/models/asl_mobilenetv2_phase1/

Environment:
    Set GCE_METADATA_IP=169.254.169.254 on GCP VM for GCS auth.
    On Kaggle: set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON.
"""

import os
import io
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from google.cloud import storage

# ── GCS config ──────────────────────────────────────────────────────────────
BUCKET = "signbridge-data"
LANDMARKS_PREFIX = "processed/asl_landmarks"
MODELS_PREFIX     = "models"

# ── Hyperparameters ──────────────────────────────────────────────────────────
MLP_CONFIG = dict(
    epochs=30, batch_size=256, lr=1e-3, patience=5, dropout=0.4
)
CNN_CONFIG = dict(
    epochs=30, batch_size=64, lr=1e-3, patience=5
)
MV2_PHASE1_CONFIG = dict(
    epochs=15, batch_size=64, lr=1e-3, patience=5
)
MV2_PHASE2_CONFIG = dict(
    epochs=15, batch_size=32, lr=1e-5, patience=5
)


# ── GCS helpers ─────────────────────────────────────────────────────────────
def _gcs_client():
    return storage.Client()


def load_npy_from_gcs(bucket_name, blob_path):
    """Download a .npy file from GCS and return as numpy array."""
    client = _gcs_client()
    blob = client.bucket(bucket_name).blob(blob_path)
    buf = io.BytesIO(blob.download_as_bytes())
    return np.load(buf)


def upload_file_to_gcs(local_path, bucket_name, blob_path):
    client = _gcs_client()
    client.bucket(bucket_name).blob(blob_path).upload_from_filename(local_path)
    print(f"  Uploaded {local_path} → gs://{bucket_name}/{blob_path}")


def save_model_to_gcs(model, model_name):
    """Save a Keras model locally then push to GCS."""
    import subprocess
    local_path = f"/tmp/{model_name}.keras"
    model.save(local_path)
    dest = f"gs://{BUCKET}/{MODELS_PREFIX}/{model_name}.keras"
    subprocess.run(["gsutil", "cp", local_path, dest], check=True)
    print(f"  Model saved to gs://{BUCKET}/{MODELS_PREFIX}/{model_name}.keras")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_landmark_data():
    """Load pre-processed ASL landmark arrays from GCS."""
    print("Loading landmark data from GCS …")
    splits = {}
    for split in ("train", "val", "test"):
        X = load_npy_from_gcs(BUCKET, f"{LANDMARKS_PREFIX}/X_{split}.npy")
        y = load_npy_from_gcs(BUCKET, f"{LANDMARKS_PREFIX}/y_{split}.npy")
        splits[split] = (X, y)
        print(f"  {split}: X{X.shape}  y{y.shape}")
    classes = load_npy_from_gcs(BUCKET, f"{LANDMARKS_PREFIX}/classes.npy")
    num_classes = len(classes)
    print(f"  Classes: {num_classes}  ({list(classes[:5])} …)")
    return splits, num_classes, classes


def load_image_data_from_gcs(image_size=224, batch_size=64):
    """
    Stream augmented images from GCS for CNN / MobileNetV2 training.
    Uses tf.data pipeline for memory efficiency.
    On Kaggle it's faster to copy images to /kaggle/working first.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # On Kaggle: images should be downloaded to LOCAL_DIR first.
    # Run:  gsutil -m cp -r gs://signbridge-data/raw/asl_alphabet/train /kaggle/working/asl_train
    LOCAL_DIR = os.environ.get("ASL_TRAIN_DIR", "/tmp/asl_train")
    if not os.path.isdir(LOCAL_DIR):
        raise RuntimeError(
            f"Image directory not found: {LOCAL_DIR}\n"
            f"Set ASL_TRAIN_DIR env var or download images first:\n"
            f"  gsutil -m cp -r gs://signbridge-data/raw/asl_alphabet/train {LOCAL_DIR}"
        )

    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        validation_split=0.125,   # 10% of 80% = ~8.7%
    )
    datagen_val = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.125)

    train_gen = datagen_train.flow_from_directory(
        LOCAL_DIR,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=42,
    )
    val_gen = datagen_val.flow_from_directory(
        LOCAL_DIR,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=42,
    )
    return train_gen, val_gen


# ── Callbacks ────────────────────────────────────────────────────────────────
def make_callbacks(ckpt_path, patience, monitor="val_loss", use_rlrop=False):
    cbs = [
        callbacks.ModelCheckpoint(
            ckpt_path, monitor=monitor, save_best_only=True, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor=monitor, patience=patience, restore_best_weights=True, verbose=1
        ),
    ]
    if use_rlrop:
        cbs.append(
            callbacks.ReduceLROnPlateau(
                monitor=monitor, factor=0.5, patience=3, min_lr=1e-7, verbose=1
            )
        )
    return cbs


# ── Evaluation helpers ───────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, classes, model_name):
    """Accuracy, macro F1, confusion matrix, inference latency."""
    from sklearn.metrics import f1_score, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n=== Evaluating {model_name} ===")

    # Accuracy
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = np.mean(y_pred == y_test)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Macro F1      : {f1:.4f}")

    # Inference latency (average over 100 single-sample predictions)
    sample = X_test[:1]
    _ = model.predict(sample, verbose=0)  # warm-up
    t0 = time.time()
    for _ in range(100):
        model.predict(sample, verbose=0)
    latency_ms = (time.time() - t0) / 100 * 1000
    print(f"  Latency (avg) : {latency_ms:.1f} ms per sample")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes,
                cmap="Blues", ax=ax, annot_kws={"size": 7})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    fig_path = f"/tmp/{model_name}_confusion.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    upload_file_to_gcs(fig_path, BUCKET, f"docs/figures/{model_name}_confusion.png")
    print(f"  Confusion matrix saved → gs://{BUCKET}/docs/figures/{model_name}_confusion.png")

    return dict(accuracy=acc, macro_f1=f1, latency_ms=latency_ms)


def plot_training_curve(history, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.suptitle(model_name)
    plt.tight_layout()
    fig_path = f"/tmp/{model_name}_training.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()
    upload_file_to_gcs(fig_path, BUCKET, f"docs/figures/{model_name}_training.png")
    print(f"  Training curve saved → gs://{BUCKET}/docs/figures/{model_name}_training.png")


# ── Model 1: Landmark MLP ────────────────────────────────────────────────────
def train_mlp():
    from src.models.cnn_classifier import build_landmark_mlp

    splits, num_classes, classes = load_landmark_data()
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)

    model = build_landmark_mlp(input_dim=63, num_classes=num_classes,
                                dropout=MLP_CONFIG["dropout"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(MLP_CONFIG["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = "/tmp/landmark_mlp_best.keras"
    cbs = make_callbacks(ckpt_path, MLP_CONFIG["patience"])

    print("\nTraining Landmark MLP …")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=MLP_CONFIG["epochs"],
        batch_size=MLP_CONFIG["batch_size"],
        callbacks=cbs,
        verbose=1,
    )
    plot_training_curve(history, "landmark_mlp")

    # Load best checkpoint for evaluation
    model = tf.keras.models.load_model(ckpt_path)
    metrics = evaluate_model(model, X_test, y_test, classes, "landmark_mlp")
    save_model_to_gcs(model, "asl_landmark_mlp_v1")
    return metrics


# ── Model 2: Baseline CNN ────────────────────────────────────────────────────
def train_baseline_cnn():
    from src.models.cnn_classifier import build_baseline_cnn

    train_gen, val_gen = load_image_data_from_gcs(batch_size=CNN_CONFIG["batch_size"])
    num_classes = train_gen.num_classes
    classes = np.array(list(train_gen.class_indices.keys()))

    model = build_baseline_cnn(num_classes=num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CNN_CONFIG["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = "/tmp/baseline_cnn_best.keras"
    cbs = make_callbacks(ckpt_path, CNN_CONFIG["patience"])

    print("\nTraining Baseline CNN …")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CNN_CONFIG["epochs"],
        callbacks=cbs,
        verbose=1,
    )
    plot_training_curve(history, "baseline_cnn")

    model = tf.keras.models.load_model(ckpt_path)

    # Load test data (landmark split reused for label indices, images used for eval)
    splits, _, _ = load_landmark_data()
    X_test, y_test = splits["test"]   # landmark features — used only for label alignment
    # For image-based eval we'd need a test image generator; report acc from val_gen instead
    print(f"\nFinal val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

    save_model_to_gcs(model, "asl_baseline_cnn_v1")
    return {"val_accuracy": max(history.history["val_accuracy"])}


# ── Model 3: MobileNetV2 (Phase 1 + Phase 2) ─────────────────────────────────
def train_mobilenetv2(phase=1, phase1_ckpt=None):
    from src.models.cnn_classifier import build_mobilenetv2, unfreeze_top_layers

    train_gen, val_gen = load_image_data_from_gcs(batch_size=MV2_PHASE1_CONFIG["batch_size"])
    num_classes = train_gen.num_classes

    if phase == 1:
        print("\n=== MobileNetV2 — Phase 1 (frozen base) ===")
        model, base = build_mobilenetv2(num_classes=num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(MV2_PHASE1_CONFIG["lr"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        ckpt_path = "/tmp/mobilenetv2_phase1_best.keras"
        cbs = make_callbacks(ckpt_path, MV2_PHASE1_CONFIG["patience"])

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=MV2_PHASE1_CONFIG["epochs"],
            callbacks=cbs,
            verbose=1,
        )
        plot_training_curve(history, "mobilenetv2_phase1")

        model = tf.keras.models.load_model(ckpt_path)
        print(f"\nPhase 1 best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
        save_model_to_gcs(model, "asl_mobilenetv2_phase1")

    else:
        print("\n=== MobileNetV2 — Phase 2 (fine-tune top 30 layers) ===")
        if phase1_ckpt and phase1_ckpt.startswith("gs://"):
            # Download from GCS
            import subprocess
            subprocess.run(
                ["gsutil", "-m", "cp", "-r", phase1_ckpt, "/tmp/mv2_phase1_dl"],
                check=True
            )
            model = tf.keras.models.load_model("/tmp/mv2_phase1_dl")
        elif phase1_ckpt:
            model = tf.keras.models.load_model(phase1_ckpt)
        else:
            raise ValueError("--phase1_ckpt required for Phase 2")

        # Unfreeze top 30 layers of MobileNetV2 base
        # The base model is the second layer in our functional model
        base_model = model.layers[2]   # MobileNetV2 is layers[2] after Input + preprocess
        unfreeze_top_layers(base_model, num_layers=30)

        train_gen_p2, val_gen_p2 = load_image_data_from_gcs(batch_size=MV2_PHASE2_CONFIG["batch_size"])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(MV2_PHASE2_CONFIG["lr"]),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        ckpt_path = "/tmp/mobilenetv2_phase2_best.keras"
        cbs = make_callbacks(ckpt_path, MV2_PHASE2_CONFIG["patience"], use_rlrop=True)

        history = model.fit(
            train_gen_p2,
            validation_data=val_gen_p2,
            epochs=MV2_PHASE2_CONFIG["epochs"],
            callbacks=cbs,
            verbose=1,
        )
        plot_training_curve(history, "mobilenetv2_phase2")

        model = tf.keras.models.load_model(ckpt_path)
        print(f"\nPhase 2 best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
        save_model_to_gcs(model, "asl_mobilenetv2_v1")   # final production model


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL models")
    parser.add_argument(
        "--model", choices=["mlp", "cnn", "mobilenetv2"], required=True,
        help="Which model to train"
    )
    parser.add_argument(
        "--phase", type=int, default=1,
        help="MobileNetV2 training phase (1 or 2)"
    )
    parser.add_argument(
        "--phase1_ckpt", type=str, default=None,
        help="Path (local or gs://) to Phase 1 checkpoint for Phase 2"
    )
    args = parser.parse_args()

    # GCP VM auth helper
    if "GCE_METADATA_IP" not in os.environ:
        os.environ["GCE_METADATA_IP"] = "169.254.169.254"

    if args.model == "mlp":
        metrics = train_mlp()
        print("\n=== MLP Final Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.model == "cnn":
        metrics = train_baseline_cnn()
        print("\n=== Baseline CNN Final Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.model == "mobilenetv2":
        train_mobilenetv2(phase=args.phase, phase1_ckpt=args.phase1_ckpt)

    print("\nDone.")
