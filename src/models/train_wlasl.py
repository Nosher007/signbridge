"""
Day 4 — WLASL Word-Level Training Script

Trains three models for comparison:
  Model 1: Landmark LSTM     (CPU-friendly, runs on GCP VM)
  Model 2: MobileNetV2+LSTM  (GPU recommended, Kaggle T4)

Step 0 (Kaggle only): Extract MobileNetV2 features from raw WLASL videos
Step 1: Train Landmark LSTM on (N, 30, 63) sequences
Step 2: Train MobileNetV2+LSTM on (N, 30, 1280) extracted features

Usage:
    # On GCP VM — landmark LSTM:
    python src/models/train_wlasl.py --model landmark_lstm

    # On Kaggle — feature extraction then MV2 LSTM:
    python src/models/train_wlasl.py --model extract_features
    python src/models/train_wlasl.py --model mobilenetv2_lstm
"""

import os
import io
import argparse
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from google.cloud import storage

# ── GCS config ───────────────────────────────────────────────────────────────
BUCKET            = "signbridge-data"
LANDMARKS_PREFIX  = "processed/wlasl_sequences"
FEATURES_PREFIX   = "processed/wlasl_mv2_features"
MODELS_PREFIX     = "models"
RAW_VIDEO_PREFIX  = "raw/wlasl/wlasl_data/videos"

# ── Hyperparameters ──────────────────────────────────────────────────────────
LSTM_CONFIG = dict(
    epochs=50, batch_size=32, lr=1e-3, patience=10, dropout=0.3
)

# ── GCS helpers ──────────────────────────────────────────────────────────────
def _client():
    return storage.Client()

def load_npy_from_gcs(blob_path):
    buf = io.BytesIO(_client().bucket(BUCKET).blob(blob_path).download_as_bytes())
    return np.load(buf)

def save_npy_to_gcs(arr, blob_path):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    _client().bucket(BUCKET).blob(blob_path).upload_from_file(buf)
    print(f"  Saved {blob_path} — shape {arr.shape}")

def save_model_to_gcs(model, model_name):
    import subprocess
    local_path = f"/tmp/{model_name}.keras"
    model.save(local_path)
    dest = f"gs://{BUCKET}/{MODELS_PREFIX}/{model_name}.keras"
    subprocess.run(["gsutil", "cp", local_path, dest], check=True)
    print(f"  Model saved → {dest}")

def upload_file_to_gcs(local_path, blob_path):
    _client().bucket(BUCKET).blob(blob_path).upload_from_filename(local_path)
    print(f"  Uploaded {local_path} → gs://{BUCKET}/{blob_path}")

# ── Data loading ─────────────────────────────────────────────────────────────
def load_landmark_data():
    """Load WLASL landmark sequences (30, 63) from GCS."""
    print("Loading WLASL landmark sequences from GCS …")
    splits = {}
    for split in ("train", "val", "test"):
        X = load_npy_from_gcs(f"{LANDMARKS_PREFIX}/X_{split}.npy")
        y = load_npy_from_gcs(f"{LANDMARKS_PREFIX}/y_{split}.npy")
        splits[split] = (X, y)
        print(f"  {split}: X{X.shape}  y{y.shape}")
    classes = load_npy_from_gcs(f"{LANDMARKS_PREFIX}/classes.npy")
    print(f"  Classes: {len(classes)}")
    return splits, len(classes), classes

def load_feature_data():
    """Load pre-extracted MobileNetV2 features (30, 1280) from GCS."""
    print("Loading WLASL MV2 features from GCS …")
    splits = {}
    for split in ("train", "val", "test"):
        X = load_npy_from_gcs(f"{FEATURES_PREFIX}/X_{split}.npy")
        y = load_npy_from_gcs(f"{FEATURES_PREFIX}/y_{split}.npy")
        splits[split] = (X, y)
        print(f"  {split}: X{X.shape}  y{y.shape}")
    classes = load_npy_from_gcs(f"{FEATURES_PREFIX}/classes.npy")
    return splits, len(classes), classes

# ── Callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks(ckpt_path, patience):
    return [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                   save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                 restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=5, min_lr=1e-7, verbose=1),
    ]

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, classes, model_name):
    from sklearn.metrics import f1_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n=== Evaluating {model_name} ===")
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Top-1 accuracy
    acc = np.mean(y_pred == y_test)

    # Top-5 accuracy
    top5 = np.mean([y_test[i] in np.argsort(y_pred_prob[i])[-5:]
                    for i in range(len(y_test))])

    # Macro F1
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"  Top-1 Accuracy : {acc*100:.2f}%")
    print(f"  Top-5 Accuracy : {top5*100:.2f}%")
    print(f"  Macro F1       : {f1:.4f}")

    # Inference latency
    sample = X_test[:1]
    _ = model.predict(sample, verbose=0)
    t0 = time.time()
    for _ in range(100):
        model.predict(sample, verbose=0)
    latency_ms = (time.time() - t0) / 100 * 1000
    print(f"  Latency (avg)  : {latency_ms:.1f} ms per sequence")

    # Confusion matrix (top 20 most common classes)
    top20 = np.argsort(np.bincount(y_test))[-20:]
    mask = np.isin(y_test, top20)
    if mask.sum() > 0:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top20)
        top20_labels = [str(classes[i]) for i in top20]
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=top20_labels,
                    yticklabels=top20_labels, cmap="Blues", ax=ax,
                    annot_kws={"size": 7})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model_name} — Confusion Matrix (Top 20 Classes)")
        plt.tight_layout()
        fig_path = f"/tmp/{model_name}_confusion.png"
        plt.savefig(fig_path, dpi=120)
        plt.close()
        upload_file_to_gcs(fig_path, f"docs/figures/{model_name}_confusion.png")

    return dict(top1=acc, top5=top5, macro_f1=f1, latency_ms=latency_ms)


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
    upload_file_to_gcs(fig_path, f"docs/figures/{model_name}_training.png")
    print(f"  Training curve → gs://{BUCKET}/docs/figures/{model_name}_training.png")


# ── Model 1: Landmark LSTM ────────────────────────────────────────────────────
def train_landmark_lstm():
    from src.models.lstm_classifier import build_landmark_lstm

    splits, num_classes, classes = load_landmark_data()
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    # Class weights to handle mild imbalance
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)

    model = build_landmark_lstm(num_classes=num_classes,
                                 dropout=LSTM_CONFIG["dropout"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LSTM_CONFIG["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = "/tmp/landmark_lstm_best.keras"
    cbs = make_callbacks(ckpt_path, LSTM_CONFIG["patience"])

    print("\nTraining Landmark LSTM …")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, tf.keras.utils.to_categorical(y_val, num_classes)),
        epochs=LSTM_CONFIG["epochs"],
        batch_size=LSTM_CONFIG["batch_size"],
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )
    plot_training_curve(history, "landmark_lstm")

    model = tf.keras.models.load_model(ckpt_path)
    metrics = evaluate_model(model, X_test, y_test, classes, "landmark_lstm")
    save_model_to_gcs(model, "wlasl_landmark_lstm_v1")
    return metrics


# ── Step 0: MobileNetV2 Feature Extraction ────────────────────────────────────
def extract_mv2_features():
    """
    Extract MobileNetV2 (1280-dim) features from raw WLASL video frames.
    Loads raw videos from GCS, decodes frames, runs through MobileNetV2.
    Saves (N, 30, 1280) arrays to GCS for each split.
    Run once on Kaggle GPU — takes ~30-60 min for 1,013 videos.
    """
    import cv2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from src.models.lstm_classifier import build_feature_extractor

    # Load landmark splits to get labels and class list
    splits_lm, num_classes, classes = load_landmark_data()

    base = build_feature_extractor()
    print(f"Feature extractor ready — output dim: 1280")

    def extract_clip_features(video_bytes, seq_len=30, image_size=224):
        """Decode video bytes → (seq_len, 1280) MV2 features."""
        arr = np.frombuffer(video_bytes, dtype=np.uint8)
        cap = cv2.VideoCapture()
        cap.open(cv2.FileStorage(arr, cv2.FILE_STORAGE_READ))

        # Write to temp file for cv2 (cv2 can't open from bytes directly)
        tmp = "/tmp/_clip.mp4"
        with open(tmp, "wb") as f:
            f.write(video_bytes)
        cap = cv2.VideoCapture(tmp)

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (image_size, image_size))
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return np.zeros((seq_len, 1280), dtype="float32")

        frames = np.array(frames, dtype="float32")  # (T, H, W, 3)

        # Even-sample or pad to seq_len
        T = len(frames)
        if T >= seq_len:
            indices = np.linspace(0, T - 1, seq_len, dtype=int)
            frames = frames[indices]
        else:
            pad = np.zeros((seq_len - T, image_size, image_size, 3), dtype="float32")
            frames = np.concatenate([frames, pad], axis=0)

        frames = preprocess_input(frames)           # (30, H, W, 3) → [-1, 1]
        features = base.predict(frames, verbose=0)  # (30, 1280)
        return features.astype("float32")

    client = _client()
    bucket = client.bucket(BUCKET)

    # List all video blobs
    all_blobs = {b.name.split("/")[-1]: b.name
                 for b in bucket.list_blobs(prefix=RAW_VIDEO_PREFIX)
                 if b.name.endswith(".mp4")}
    print(f"Found {len(all_blobs)} video files in GCS")

    # For each split, we need to match video filenames to the sequences
    # The landmark preprocessing stored sequences by index — we need to re-extract
    # using the same video list order. Load the class mapping from landmark split.
    # Since we don't have video filenames stored, re-run extraction for all videos
    # and create fresh train/val/test splits matching landmark split labels.

    # Simpler approach: extract features for all videos, split same way as landmarks
    all_features = []
    all_labels   = []

    # Get full video list from GCS (same order as preprocessing)
    video_blobs = sorted(all_blobs.keys())
    class_to_idx = {str(c): i for i, c in enumerate(classes)}

    print(f"Extracting features from {len(video_blobs)} videos …")
    for i, vname in enumerate(video_blobs):
        blob_path = all_blobs[vname]
        # Infer class from parent folder name
        parts = blob_path.split("/")
        # Path: raw/wlasl/wlasl_data/videos/<gloss>/<file>.mp4
        gloss = parts[-2] if len(parts) >= 2 else "unknown"
        if gloss not in class_to_idx:
            continue

        video_bytes = bucket.blob(blob_path).download_as_bytes()
        feat = extract_clip_features(video_bytes)
        all_features.append(feat)
        all_labels.append(class_to_idx[gloss])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(video_blobs)} done")

    all_features = np.array(all_features, dtype="float32")  # (N, 30, 1280)
    all_labels   = np.array(all_labels,   dtype="int32")

    print(f"Total extracted: {all_features.shape}")

    # Split same ratios as landmark split (70/10/10/10)
    from sklearn.model_selection import train_test_split
    N = len(all_features)
    idx = np.arange(N)
    idx_train, idx_temp = train_test_split(idx, test_size=0.3, random_state=42)
    idx_val,   idx_test  = train_test_split(idx_temp, test_size=0.5, random_state=42)

    splits_out = {
        "train": (all_features[idx_train], all_labels[idx_train]),
        "val":   (all_features[idx_val],   all_labels[idx_val]),
        "test":  (all_features[idx_test],  all_labels[idx_test]),
    }

    for split, (X, y) in splits_out.items():
        save_npy_to_gcs(X, f"{FEATURES_PREFIX}/X_{split}.npy")
        save_npy_to_gcs(y, f"{FEATURES_PREFIX}/y_{split}.npy")
    save_npy_to_gcs(classes, f"{FEATURES_PREFIX}/classes.npy")

    print("Feature extraction complete.")
    return splits_out


# ── Model 2: MobileNetV2 + LSTM ───────────────────────────────────────────────
def train_mobilenetv2_lstm():
    from src.models.lstm_classifier import build_mobilenetv2_lstm

    splits, num_classes, classes = load_feature_data()
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)

    model = build_mobilenetv2_lstm(num_classes=num_classes,
                                    dropout=LSTM_CONFIG["dropout"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LSTM_CONFIG["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = "/tmp/mobilenetv2_lstm_best.keras"
    cbs = make_callbacks(ckpt_path, LSTM_CONFIG["patience"])

    print("\nTraining MobileNetV2+LSTM …")
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=LSTM_CONFIG["epochs"],
        batch_size=LSTM_CONFIG["batch_size"],
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )
    plot_training_curve(history, "mobilenetv2_lstm")

    model = tf.keras.models.load_model(ckpt_path)
    metrics = evaluate_model(model, X_test, y_test, classes, "mobilenetv2_lstm")
    save_model_to_gcs(model, "wlasl_mobilenetv2_lstm_v1")
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WLASL models")
    parser.add_argument(
        "--model",
        choices=["landmark_lstm", "extract_features", "mobilenetv2_lstm"],
        required=True,
    )
    args = parser.parse_args()

    if "GCE_METADATA_IP" not in os.environ:
        os.environ["GCE_METADATA_IP"] = "169.254.169.254"

    if args.model == "landmark_lstm":
        metrics = train_landmark_lstm()
        print("\n=== Landmark LSTM Final Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    elif args.model == "extract_features":
        extract_mv2_features()

    elif args.model == "mobilenetv2_lstm":
        metrics = train_mobilenetv2_lstm()
        print("\n=== MobileNetV2+LSTM Final Metrics ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    print("\nDone.")
