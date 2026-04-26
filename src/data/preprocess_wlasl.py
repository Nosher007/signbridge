"""
WLASL preprocessing pipeline.
Reads video clips from GCS, extracts per-frame MediaPipe landmarks,
pads/truncates to 30 frames, saves sequences to GCS.

Output: gs://signbridge-data/processed/wlasl_sequences/
  - X_train.npy  shape (N_train, 30, 63)
  - X_val.npy    shape (N_val, 30, 63)
  - X_test.npy   shape (N_test, 30, 63)
  - y_train.npy, y_val.npy, y_test.npy  (integer class labels)
  - classes.npy  (sorted gloss list)
"""

import os
import io
import json
import numpy as np
from google.cloud import storage
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.mediapipe_extractor import extract_from_video_bytes, get_hands_model

BUCKET = "signbridge-data"
WLASL_PREFIX = "raw/wlasl/wlasl_data/"
VIDEO_PREFIX = "raw/wlasl/wlasl_data/videos/"
OUTPUT_PREFIX = "processed/wlasl_sequences/"
PROJECT = "signbridge-prod"

SEQUENCE_LENGTH = 30
VAL_SIZE = 0.10 / 0.80
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

    # Load top-100 split and full annotations
    print("Loading WLASL annotations...")
    nslt_100 = json.loads(bucket.blob(WLASL_PREFIX + "nslt_100.json").download_as_text())
    wlasl_data = json.loads(bucket.blob(WLASL_PREFIX + "WLASL_v0.3.json").download_as_text())
    top100_ids = set(nslt_100.keys())

    # Build video_id → gloss mapping
    vid_to_gloss = {}
    for entry in wlasl_data:
        for inst in entry["instances"]:
            vid_id = str(inst["video_id"])
            if vid_id in top100_ids:
                vid_to_gloss[vid_id] = entry["gloss"]

    classes = sorted(set(vid_to_gloss.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes[:10]} ...")

    # Get available videos in GCS
    available = set(
        b.name.split("/")[-1].replace(".mp4", "")
        for b in client.list_blobs(BUCKET, prefix=VIDEO_PREFIX)
        if b.name.endswith(".mp4")
    )
    # Filter to top-100 videos that exist
    video_ids = [vid for vid in top100_ids if vid in available]
    print(f"Videos available in GCS: {len(available)}, in top-100: {len(video_ids)}")

    X, y = [], []
    failed = 0
    with get_hands_model() as hands:
        for i, vid_id in enumerate(video_ids):
            gloss = vid_to_gloss.get(vid_id)
            if gloss is None:
                continue
            try:
                blob_path = VIDEO_PREFIX + vid_id + ".mp4"
                vid_bytes = bucket.blob(blob_path).download_as_bytes()
                seq = extract_from_video_bytes(vid_bytes, hands, SEQUENCE_LENGTH)
                X.append(seq)
                y.append(class_to_idx[gloss])
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Warning: failed on {vid_id}: {e}")
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(video_ids)} videos...")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"\nFull dataset: X={X.shape}, y={y.shape}, failed={failed}")
    assert X.shape[1:] == (SEQUENCE_LENGTH, 63), f"Expected (N, 30, 63), got {X.shape}"
    assert not np.any(np.isnan(X)), "NaN values found in sequences!"

    # 70/10/10/10 split
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

    print("\npreprocess_wlasl.py complete.")


if __name__ == "__main__":
    main()
