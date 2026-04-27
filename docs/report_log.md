# SignBridge — Living Report Log

> Fill this in after every meaningful step. Day 7 compiles this into the final PDF.
> Rule: if it's not here, it doesn't go in the report.

---

<!-- EDA findings, model runs, decisions, and LLM evaluations go below -->

## EDA Finding — ASL Alphabet Dataset — 2026-04-26

**What we looked at:** Class distribution, image sizes, and sample visualization of the Kaggle ASL Alphabet dataset.

**What we found:**
- 29 classes total: 26 letters (A–Z) + `del`, `nothing`, `space`
- Class distribution is perfectly uniform — exactly 3,000 images per class (87,000 total)
- All images are 200×200px with clean, mostly white backgrounds
- J and Z are included as static approximations despite normally requiring motion
- No class imbalance — class weighting not needed for training

**Figure saved:** `docs/figures/asl_class_dist.png`, `docs/figures/asl_samples.png`

**Impact on plan:**
Resize all images from 200×200 to 224×224 for MobileNetV2 input. Apply background augmentation (random backgrounds, brightness jitter, rotation) to improve real-world webcam generalization. No class weighting needed. J and Z may underperform at inference since real signing requires motion — worth noting as a known limitation.

---

## EDA Finding — WLASL Dataset (Top-100 Words) — 2026-04-26

**What we looked at:** Clip count distribution, frame count distribution, signer diversity, and MediaPipe landmark coverage on a sample video.

**What we found:**
- 100 unique glosses, 2,038 total clips in the top-100 subset
- Clip counts range from 18 to 40 per gloss (mean: 20.4) — mild class imbalance present
- Frame counts: Min 26, Max 149, Mean 71.7, **Median 66** → clips are much longer than our target 30 frames
- Average 15 unique signers per gloss (range: 10–21) — good signer diversity for generalization
- 11,980 video files confirmed in GCS
- MediaPipe detected hand landmarks in **41.4% of frames** on sample video (`00335.mp4`, 58 frames total, 24 detected)
- Landmark shape confirmed: **(21, 3)** per frame ✓

**Figure saved:** `docs/figures/wlasl_clip_dist.png`, `docs/figures/wlasl_frame_dist.png`, `docs/figures/wlasl_signer_diversity.png`

**Impact on plan:**
- Pad/truncate all sequences to **30 frames** during preprocessing (median is 66 — we sample evenly or take the first 30)
- Apply **class weights** during LSTM training to handle the 18–40 clip imbalance
- **Zero-fill** frames where MediaPipe fails to detect a hand (41% coverage on sample is low — may need detection confidence tuning or fallback strategy)
- The low MediaPipe coverage is a known WLASL challenge: videos come from YouTube with varied lighting, angles, and backgrounds. This will be noted as a limitation in the report.

---

## Preprocessing Decision — Day 2 — 2026-04-26

**What we built:** MediaPipe landmark extraction pipeline for both ASL images and WLASL videos.

**Key decisions:**

**Landmark extraction:**
- Used MediaPipe Hands with `max_num_hands=1`, `min_detection_confidence=0.7`
- Extract 21 landmarks × 3 coords (x, y, z) = **63 features per frame**
- Zero-fill frames where no hand is detected (rather than dropping)

**ASL preprocessing:**
- Bulk-downloaded all 87,000 images to VM local disk first (`gsutil -m`), then ran MediaPipe from disk — ~20× faster than per-image GCS requests
- Output: `(N, 63)` arrays per split
- Split: Train(60,900) / Val(8,700) / Test(8,700) / LLM(8,700) — stratified by class

**WLASL preprocessing:**
- Extracted per-frame landmarks from all video clips via GCS streaming
- Sequences sampled evenly to exactly **30 frames** (median video was 66 frames)
- Shorter videos zero-padded at the end
- Output: `(N, 30, 63)` arrays per split
- Split: Train(708) / Val(102) / Test(101) / LLM(102) — random (some classes had <2 clips, stratify not possible)

**Validation results:**
- ASL: all 4 splits pass shape and NaN checks ✓
- WLASL: all 4 splits pass shape and NaN checks ✓
- 29 ASL classes and 100 WLASL glosses confirmed

---

## Model Run — Landmark MLP — ASL Alphabet — 2026-04-27

**Architecture:**
3-layer fully connected network (MLP) trained on 63 MediaPipe landmark features per frame. No image pixels — purely hand geometry. ~60K parameters. Lower-bound reference: tests whether hand skeleton alone can distinguish letters.

**Hyperparameters:**
- Epochs: 25 (early stopping at epoch 25, patience=5)
- Batch size: 256
- Learning rate: 1e-3 (Adam)
- Dropout: 0.4
- Architecture: Dense(256,relu) → BN → Dropout → Dense(128,relu) → BN → Dropout → Dense(64,relu) → Dropout → Dense(29,softmax)

**Results:**
| Metric            | Value    |
|-------------------|----------|
| Test Accuracy     | 59.01%   |
| Macro F1          | 0.6837   |
| Top-5 Accuracy    | N/A      |
| Inference Latency | 63.1 ms  |
| Parameters        | ~0.06M   |

**Training curve figure:** `docs/figures/landmark_mlp_training.png`
**Confusion matrix figure:** `docs/figures/landmark_mlp_confusion.png`

**Observations:**
- Loss decreased steadily from epoch 1 (2.42) to plateau around epoch 20–25
- Model plateaued at ~59–60% accuracy — consistent with known limitation: 63 landmark features struggle to distinguish visually similar letter pairs (M/N, G/H, R/U) where hand shape differences are subtle in 3D joint coordinates
- Macro F1 of 0.6837 is higher than accuracy because some classes reach high precision/recall while confusable classes drag accuracy down
- 63.1 ms/sample latency is well within real-time threshold but is based purely on CPU MLP forward pass

**Decision:**
Proceed to next baseline (CNN from scratch). This MLP result is the lower-bound reference — expected to be outperformed by image-based models with richer spatial features.
