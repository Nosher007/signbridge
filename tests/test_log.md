# SignBridge — Test Log

> Every task gets a test. Log results here before marking a task done.

---

## Day 1 — Setup + Data + EDA

- [x] GCP setup → `gsutil ls gs://signbridge-data/` ✅ — project `signbridge-prod` created, billing linked, 6 APIs enabled, bucket confirmed
- [x] GCP VM setup ✅ — `signbridge-vm` (n2-standard-4, us-central1-a) running, SSH confirmed, GCS accessible from VM. Training on Kaggle free T4 (GPU exhausted globally); CPU VM used for data/EDA work.
- [x] ASL dataset upload → 87,000 images confirmed in GCS (`raw/asl_alphabet/train/`, 29 class folders) ✅
- [x] WLASL dataset upload → 11,980 video clips confirmed in GCS (`raw/wlasl/wlasl_data/videos/`) ✅
- [x] ASL EDA notebook → all cells ran, class dist plot + samples saved, 29 classes × 3,000 images each ✅
- [x] WLASL EDA notebook → all cells ran, clip dist + frame dist + signer diversity plots saved, MediaPipe (21,3) ✓ ✅
- [x] `pip install -r requirements.txt` → no errors on GCP VM ✅
- [x] GitHub repo initial commit → pushed to https://github.com/Nosher007/signbridge ✅

## Day 2 — Preprocessing + MediaPipe Pipeline

- [x] `mediapipe_extractor.py` → image shape (63,) ✓, video shape (30, 63) ✓ ✅
- [x] `preprocess_asl.py` → 87k images processed, splits saved to GCS ✅
- [x] `preprocess_wlasl.py` → 1,013 videos processed, splits saved to GCS ✅
- [x] `validate_processed.py` → ASL PASSED, WLASL PASSED, 0 NaNs ✅
- [x] ASL processed shape: Train(60900,63) Val(8700,63) Test(8700,63) LLM(8700,63) ✅
- [x] WLASL processed shape: Train(708,30,63) Val(102,30,63) Test(101,30,63) LLM(102,30,63) ✅

## Day 3 — ASL CNN Model Training (in progress)

- [x] `cnn_classifier.py` → 3 model architectures defined: LandmarkMLP, BaselineCNN, MobileNetV2 ✅
- [x] `train_asl.py` → training script with GCS data loading, callbacks, eval, model upload ✅
- [x] Landmark MLP trained on CPU VM → loss ↓ epochs 1–25, no NaN loss ✅
- [x] Landmark MLP test accuracy: **59.01%**, Macro F1: **0.6837**, Latency: **63.1 ms** ✅
- [x] Landmark MLP confusion matrix + training curve saved to GCS `docs/figures/` ✅
- [x] Landmark MLP best checkpoint saved to `gs://signbridge-data/models/asl_landmark_mlp_v1.keras` ✅
- [ ] Baseline CNN trained on Kaggle T4 — pending
- [ ] MobileNetV2 Phase 1 trained on Kaggle T4 — pending
- [ ] MobileNetV2 Phase 2 fine-tuned on Kaggle T4 — pending
