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
