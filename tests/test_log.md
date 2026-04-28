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
- [x] Baseline CNN trained on Kaggle T4 → val accuracy 99.99%, val loss 0.00063, early stop epoch 15 ✅
- [x] MobileNetV2 Phase 1 trained on Kaggle T4 → val accuracy 99.80%, val loss 0.0072, early stop epoch 15 ✅
- [x] MobileNetV2 Phase 2 fine-tuned on Kaggle T4 → val accuracy 99.99%, val loss 0.00018, epoch 15 ✅
- [x] All models saved to GCS → `gs://signbridge-data/models/` ✅
- [x] Loss decreased every epoch in Phase 2 — no NaN loss ✅
- [x] MobileNetV2 selected as final ASL model ✅

## Day 4 — WLASL LSTM Model Training

- [x] `lstm_classifier.py` → build_landmark_lstm, build_mobilenetv2_lstm, build_feature_extractor defined ✅
- [x] `train_wlasl.py` → training script with GCS data loading, callbacks, eval, model upload ✅
- [x] `04_train_wlasl_kaggle.ipynb` → 8-cell Kaggle notebook for T4 GPU training ✅
- [x] MobileNetV2 feature extraction → (30, 1280) per clip confirmed ✅
- [x] Feature extraction complete → Train(748), Val(165), Test(100) clips extracted; 1,025 clips unavailable (YouTube takedowns, expected) ✅
- [x] Features saved to GCS → `processed/wlasl_mv2_features/X_train.npy`, `X_val.npy`, `X_test.npy` ✅
- [x] LSTM training → loss decreased over training, no NaN loss ✅
- [x] MobileNetV2+LSTM final results: Top-1: **9.00%**, Top-5: **23.00%**, Macro F1: **0.0559**, Latency: **85.2 ms** ✅
- [x] Model saved to GCS → `gs://signbridge-data/models/wlasl_mobilenetv2_lstm_v1.keras` ✅
- [x] Low accuracy expected and documented: ~7-8 clips/class after video attrition (vs ~21/class in full WLASL benchmark) ✅
- [x] MobileNetV2+LSTM selected as final WLASL model — outperforms landmark LSTM on Top-5 ✅
- [x] Day 4 model run entries + model decision entry logged to `docs/report_log.md` ✅

## Day 4 Extension — ASL-Citizen Augmentation Experiment

- [x] ASL-Citizen dataset attached to Kaggle notebook (abd0kamel/asl-citizen, 83,399 videos) ✅
- [x] Dataset path confirmed: `/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen` ✅
- [x] Splits loaded from `splits/train.csv`, `splits/val.csv`, `splits/test.csv` ✅
- [x] Overlap with WLASL top-100: **67 glosses**, 2,050 clips (avg 30.6/gloss), 0 failed ✅
- [x] MobileNetV2 features extracted for all 2,050 ASL-Citizen clips — 0 failures ✅
- [x] Merged features saved to GCS → Train(1721) / Val(423) / Test(919) ✅
- [x] Experiment 1 — Mixed WLASL+ASL-Citizen, 100 classes → Val accuracy: **4.02%** (worse than v1) ✅
- [x] Experiment 2 — Feature normalization (StandardScaler) → Val accuracy: **3.55%** (no improvement) ✅
- [x] Experiment 3 — ASL-Citizen only, 100 classes → Val accuracy: **3.10%** (ghost class problem) ✅
- [x] Experiment 4 — ASL-Citizen only, 67 active classes → Val accuracy: **2.33%** (signer-independent evaluation too hard) ✅
- [x] Root cause documented: domain mismatch + signer-independent evaluation + insufficient data per class ✅
- [x] Decision: revert to WLASL-only v1 model (9% Top-1, 23% Top-5) as production model ✅
- [x] Full experiment log added to `docs/report_log.md` ✅
- [x] GCP VM stopped to preserve billing ✅

## Day 5 — LangChain + Gemini Pipeline

- [x] `langchain_pipeline.py` built — `SignBridgePipeline` class with `translate()` + `translate_batch()` ✅
- [x] Gemini API auth confirmed — `ChatGoogleGenerativeAI` with Google AI Studio key ✅
- [x] Model: `gemini-2.5-flash` (gemini-2.0-flash deprecated for new users) ✅
- [x] Pipeline run on `["HELLO", "MY", "NAME", "IS", "N", "O", "S", "H"]` → "Hello, my name is Nosh." ✅
- [x] Pipeline run on `["HELP", "M", "E"]` → "Help me." ✅
- [x] All 10 test sequences → 100% success rate, 0 API failures ✅
- [x] Avg quality score: 4.7 / 5 ✅
- [x] Avg latency: 1112 ms ✅
- [x] Retry logic in place (exponential backoff: 2s, 4s) ✅
- [x] LLM Pipeline Finding entry logged to `docs/report_log.md` ✅
