"""
Validation script — confirms processed data in GCS is correct.
Run after preprocess_asl.py and preprocess_wlasl.py complete.
"""

import io
import numpy as np
from google.cloud import storage

BUCKET = "signbridge-data"
PROJECT = "signbridge-prod"


def load_npy(bucket_obj, path):
    data = bucket_obj.blob(path).download_as_bytes()
    return np.load(io.BytesIO(data), allow_pickle=True)


def validate_asl(bucket):
    print("=== ASL Landmark Validation ===")
    prefix = "processed/asl_landmarks/"
    splits = ["train", "val", "test", "llm"]
    for split in splits:
        X = load_npy(bucket, prefix + f"X_{split}.npy")
        y = load_npy(bucket, prefix + f"y_{split}.npy")
        assert X.ndim == 2 and X.shape[1] == 63, f"Bad shape: {X.shape}"
        assert len(X) == len(y), "X/y length mismatch"
        assert not np.any(np.isnan(X)), f"NaN in X_{split}"
        print(f"  {split}: X={X.shape}, y={y.shape}, NaN=0 ✓")
    classes = load_npy(bucket, prefix + "classes.npy")
    print(f"  Classes: {len(classes)} → {list(classes)}")
    print("ASL validation PASSED\n")


def validate_wlasl(bucket):
    print("=== WLASL Sequence Validation ===")
    prefix = "processed/wlasl_sequences/"
    splits = ["train", "val", "test", "llm"]
    for split in splits:
        X = load_npy(bucket, prefix + f"X_{split}.npy")
        y = load_npy(bucket, prefix + f"y_{split}.npy")
        assert X.ndim == 3 and X.shape[1] == 30 and X.shape[2] == 63, \
            f"Bad shape: {X.shape}"
        assert len(X) == len(y), "X/y length mismatch"
        assert not np.any(np.isnan(X)), f"NaN in X_{split}"
        print(f"  {split}: X={X.shape}, y={y.shape}, NaN=0 ✓")
    classes = load_npy(bucket, prefix + "classes.npy")
    print(f"  Classes: {len(classes)} → {list(classes[:5])} ...")
    print("WLASL validation PASSED\n")


if __name__ == "__main__":
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(BUCKET)
    validate_asl(bucket)
    validate_wlasl(bucket)
    print("All validations PASSED ✓")
