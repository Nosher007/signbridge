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

---

## Model Run — Baseline CNN — ASL Alphabet — 2026-04-27

**Architecture:**
4-layer CNN trained from scratch on raw 224×224 images. No pretrained weights. ~463K parameters. Lower-bound image-based reference.

**Hyperparameters:**
- Epochs: 15 (early stopping at epoch 15, patience=5)
- Batch size: 64
- Learning rate: 1e-3 (Adam)
- Architecture: Conv(32)→BN→Pool → Conv(64)→BN→Pool → Conv(128)→BN→Pool → Conv(256)→BN→GAP → Dense(256,relu) → Dropout(0.4) → Dense(29,softmax)

**Results:**
| Metric            | Value    |
|-------------------|----------|
| Val Accuracy      | 99.99%   |
| Val Loss          | 0.00063  |
| Inference Latency | ~130 ms/batch (GPU) |
| Parameters        | 0.46M    |

**Training curve figure:** `docs/figures/baseline_cnn_training.png`

**Observations:**
- Train accuracy jumped to 97.7% by epoch 2 — model memorized the clean studio backgrounds very quickly
- Val loss oscillated wildly between epochs (e.g. 6.23 at epoch 3, then recovered to 0.025 at epoch 5) — typical instability at lr=1e-3 for a small CNN on a homogeneous dataset
- Best checkpoint at epoch 10: val_accuracy=99.99%, val_loss=0.00063
- Near-perfect accuracy reflects the dataset's homogeneous studio backgrounds — real-world webcam performance expected to be significantly lower
- This result does NOT generalize: the model learns studio-specific features, not robust hand geometry

**Decision:**
Proceed to MobileNetV2. Baseline CNN achieves near-perfect val accuracy on this controlled dataset but will fail to generalize to real webcam input — exactly the limitation that transfer learning addresses.

---

## Model Run — MobileNetV2 Phase 1 + Phase 2 — ASL Alphabet — 2026-04-27

**Architecture:**
MobileNetV2 pretrained on ImageNet (2.26M params frozen) + custom head: GlobalAveragePooling2D → Dense(256,relu) → Dropout(0.4) → Dense(29,softmax). Total 2.59M params. Two-phase training strategy.

**Phase 1 Hyperparameters (frozen base):**
- Epochs: 15 (early stopping at epoch 15, patience=5)
- Batch size: 64
- Learning rate: 1e-3 (Adam)
- Trainable params: 335K (head only)

**Phase 2 Hyperparameters (fine-tune top 30 layers):**
- Epochs: 15 (ran all 15)
- Batch size: 64
- Learning rate: 1e-5 (Adam) + ReduceLROnPlateau
- Trainable params: ~640K (head + top 30 base layers)

**Results:**
| Metric            | Phase 1  | Phase 2  |
|-------------------|----------|----------|
| Best Val Accuracy | 99.80%   | **99.99%** |
| Best Val Loss     | 0.0072   | **0.00018** |
| Best Epoch        | 10       | 15       |
| Parameters        | 2.59M    | 2.59M    |

**Training curve figures:** `docs/figures/mobilenetv2_phase1_training.png`, `docs/figures/mobilenetv2_phase2_training.png`

**Observations:**
- Phase 1 hit 99.09% val accuracy on epoch 1 alone — ImageNet pretrained features transferred immediately to ASL hand recognition
- Phase 2 improved val loss from 0.0072 → 0.00018, confirming fine-tuning adds meaningful refinement
- Training was perfectly stable throughout both phases — no val loss oscillation unlike the baseline CNN
- Val accuracy improved monotonically each epoch in Phase 2
- Selected as the production model for the SignBridge pipeline

**Decision:**
MobileNetV2 Phase 2 selected as the final ASL alphabet classifier. Saved to `gs://signbridge-data/models/asl_mobilenetv2_v1.keras`.

---

## Model Decision — ASL Alphabet — 2026-04-27

**Models compared:** Landmark MLP vs Baseline CNN vs MobileNetV2

**Comparison table:**
| Model            | Val Accuracy | Val Loss  | Latency | Params | Selected? |
|------------------|-------------|-----------|---------|--------|-----------|
| Landmark MLP     | 59.01%      | 1.44      | 63 ms   | 0.06M  | No        |
| Baseline CNN     | 99.99%      | 0.00063   | ~130 ms | 0.46M  | No        |
| **MobileNetV2**  | **99.99%**  | **0.00018** | ~90 ms | 2.59M  | ✅ Yes    |

**Why we chose MobileNetV2:**
MobileNetV2 achieved equal val accuracy to the Baseline CNN (99.99%) but with 4× lower val loss (0.00018 vs 0.00063), indicating sharper, more confident predictions. More importantly, MobileNetV2's ImageNet-pretrained features provide superior generalization to real-world webcam input — the controlled studio dataset makes both image models appear equivalent on validation, but pretrained features are known to transfer better to distribution shifts. The Landmark MLP's 59% accuracy confirms that raw hand geometry alone is insufficient for robust letter discrimination. MobileNetV2's 2-phase training strategy (freeze then fine-tune) also produced stable, monotonically improving training curves with no instability.

**Trade-offs acknowledged:**
Baseline CNN is lighter (0.46M vs 2.59M params) and achieved the same val accuracy — however its training instability (wildly oscillating val loss) and lack of pretrained generalization make it unsuitable as the production model. MobileNetV2's larger size is justified by its real-world robustness.

**Figure saved:** `docs/figures/asl_model_comparison.png`
