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

"""
Fast approach: bulk-download all images to local disk first using gsutil -m,
then process from disk (100x faster than per-image GCS requests).
"""

import os
import io
import subprocess
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.mediapipe_extractor import extract_from_image, get_hands_model

BUCKET = "signbridge-data"
TRAIN_PREFIX = "raw/asl_alphabet/train/asl_alphabet_train/"
OUTPUT_PREFIX = "processed/asl_landmarks/"
PROJECT = "signbridge-prod"
LOCAL_DIR = "/tmp/asl_train"

VAL_SIZE = 0.10 / 0.80
RANDOM_SEED = 42


def upload_npy(bucket_obj, array, gcs_path):
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    blob = bucket_obj.blob(gcs_path)
    blob.upload_from_file(buf, content_type="application/octet-stream")
    print(f"  Saved {gcs_path} — shape {array.shape}", flush=True)


def main():
    # Step 1: Bulk download all images to local disk
    print("Step 1: Downloading all ASL images to local disk...", flush=True)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    cmd = f"gsutil -m cp -r gs://{BUCKET}/{TRAIN_PREFIX}* {LOCAL_DIR}/"
    print(f"  Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)
    print("  Download complete.", flush=True)

    # Step 2: Discover classes from local disk
    classes = sorted([
        d for d in os.listdir(LOCAL_DIR)
        if os.path.isdir(os.path.join(LOCAL_DIR, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"\nStep 2: Found {len(classes)} classes: {classes}", flush=True)

    # Step 3: Extract landmarks from local images
    print("\nStep 3: Extracting MediaPipe landmarks from local images...", flush=True)
    X, y = [], []
    with get_hands_model() as hands:
        for cls in classes:
            cls_dir = os.path.join(LOCAL_DIR, cls)
            images = [f for f in os.listdir(cls_dir) if f.endswith(".jpg")]
            print(f"  {cls}: {len(images)} images", flush=True)
            for img_file in images:
                img_path = os.path.join(cls_dir, img_file)
                lm = extract_from_image(img_path, hands)
                X.append(lm)
                y.append(class_to_idx[cls])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nFull dataset: X={X.shape}, y={y.shape}", flush=True)
    assert X.shape[1] == 63
    assert not np.any(np.isnan(X)), "NaN values found!"

    # Step 4: Split
    X_trainval, X_held, y_trainval, y_held = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_test, X_llm, y_test, y_llm = train_test_split(
        X_held, y_held, test_size=0.50, random_state=RANDOM_SEED, stratify=y_held
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_trainval
    )
    print(f"\nSplits — Train:{X_train.shape} Val:{X_val.shape} Test:{X_test.shape} LLM:{X_llm.shape}", flush=True)

    # Step 5: Upload to GCS
    print("\nStep 5: Uploading to GCS...", flush=True)
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    upload_npy(bucket, X_train,  OUTPUT_PREFIX + "X_train.npy")
    upload_npy(bucket, y_train,  OUTPUT_PREFIX + "y_train.npy")
    upload_npy(bucket, X_val,    OUTPUT_PREFIX + "X_val.npy")
    upload_npy(bucket, y_val,    OUTPUT_PREFIX + "y_val.npy")
    upload_npy(bucket, X_test,   OUTPUT_PREFIX + "X_test.npy")
    upload_npy(bucket, y_test,   OUTPUT_PREFIX + "y_test.npy")
    upload_npy(bucket, X_llm,    OUTPUT_PREFIX + "X_llm.npy")
    upload_npy(bucket, y_llm,    OUTPUT_PREFIX + "y_llm.npy")
    upload_npy(bucket, np.array(classes), OUTPUT_PREFIX + "classes.npy")

    print("\npreprocess_asl.py complete.", flush=True)


if __name__ == "__main__":
    main()
