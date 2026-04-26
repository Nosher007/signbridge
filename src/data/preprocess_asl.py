"""
ASL Alphabet preprocessing pipeline.
Reads raw images from GCS, extracts MediaPipe landmarks,
saves processed landmark arrays back to GCS.

Output: gs://signbridge-data/processed/asl_landmarks/
  - X_train.npy  shape (N_train, 63)
  - X_val.npy    shape (N_val, 63)
  - X_test.npy   shape (N_test, 63)
  - y_train.npy, y_val.npy, y_test.npy  (integer class labels)
  - classes.npy  (sorted class name list)
"""

import os
import io
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
import sys

# Allow importing from src/pipeline
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.mediapipe_extractor import extract_from_image_bytes, get_hands_model

BUCKET = "signbridge-data"
TRAIN_PREFIX = "raw/asl_alphabet/train/asl_alphabet_train/"
OUTPUT_PREFIX = "processed/asl_landmarks/"
PROJECT = "signbridge-prod"

VAL_SIZE = 0.10 / 0.80   # 10% of train+val split
RANDOM_SEED = 42


def upload_npy(bucket_obj, array, gcs_path):
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    blob = bucket_obj.blob(gcs_path)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    print(f"  Saved {gcs_path} — shape {array.shape}")


def main():
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)

    # Discover classes
    blobs = client.list_blobs(BUCKET, prefix=TRAIN_PREFIX, delimiter="/")
    _ = list(blobs)
    classes = sorted([p.rstrip("/").split("/")[-1] for p in blobs.prefixes])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    X, y = [], []
    with get_hands_model() as hands:
        for cls in classes:
            prefix = TRAIN_PREFIX + cls + "/"
            blobs_cls = list(client.list_blobs(BUCKET, prefix=prefix))
            print(f"  Processing {cls}: {len(blobs_cls)} images")
            for blob in blobs_cls:
                img_bytes = blob.download_as_bytes()
                lm = extract_from_image_bytes(img_bytes, hands)
                X.append(lm)
                y.append(class_to_idx[cls])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nFull dataset: X={X.shape}, y={y.shape}")
    assert X.shape[1] == 63, f"Expected 63 features, got {X.shape[1]}"
    assert not np.any(np.isnan(X)), "NaN values found in landmarks!"

    # 70/10/10/10 split — hold out 20% first (test + llm_eval), then split train/val
    X_trainval, X_held, y_trainval, y_held = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_test, X_llm, y_test, y_llm = train_test_split(
        X_held, y_held, test_size=0.50, random_state=RANDOM_SEED, stratify=y_held
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_trainval
    )

    print(f"\nSplit sizes:")
    print(f"  Train:    {X_train.shape}")
    print(f"  Val:      {X_val.shape}")
    print(f"  CV Test:  {X_test.shape}")
    print(f"  LLM Eval: {X_llm.shape}")

    # Upload to GCS
    print("\nUploading to GCS...")
    upload_npy(bucket, X_train,  OUTPUT_PREFIX + "X_train.npy")
    upload_npy(bucket, y_train,  OUTPUT_PREFIX + "y_train.npy")
    upload_npy(bucket, X_val,    OUTPUT_PREFIX + "X_val.npy")
    upload_npy(bucket, y_val,    OUTPUT_PREFIX + "y_val.npy")
    upload_npy(bucket, X_test,   OUTPUT_PREFIX + "X_test.npy")
    upload_npy(bucket, y_test,   OUTPUT_PREFIX + "y_test.npy")
    upload_npy(bucket, X_llm,    OUTPUT_PREFIX + "X_llm.npy")
    upload_npy(bucket, y_llm,    OUTPUT_PREFIX + "y_llm.npy")
    upload_npy(bucket, np.array(classes), OUTPUT_PREFIX + "classes.npy")

    print("\npreprocess_asl.py complete.")


if __name__ == "__main__":
    main()
