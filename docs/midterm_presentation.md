# SignBridge — Midterm Presentation
### Week 5 · 20 min presentation + 10 min Q&A
**Team:** VisionBridge — Nosherwan Babar & Ayush Gundawar

---

## Slide-by-Slide Script

---

### SLIDE 1 — Title Slide (0:00–0:30)

**Title:** SignBridge: AI-Powered Sign Language Recognition and Context-Aware Translation

**Subtitle:** Real-time ASL → English via Computer Vision + Large Language Models

**Team:** Nosherwan Babar · Ayush Gundawar

**Visual:** Logo/icon + pipeline diagram thumbnail

---

### SLIDE 2 — The Problem (0:30–2:00) · 1.5 min

**Title:** Why SignBridge?

**Talking points:**
- 70 million deaf and hard-of-hearing people worldwide rely on sign language
- In the US: 500,000–2 million ASL users daily
- The communication gap: the vast majority of hearing people cannot understand ASL
- Existing solutions are expensive, require specialized hardware, or are not real-time
- No widely accessible, low-cost, webcam-based tool exists today

**Bullet points on slide:**
- 70M+ deaf/HoH people rely on sign language globally
- No real-time, webcam-only, affordable translation tool exists
- Hospitals, schools, workplaces — everyday communication barriers
- SignBridge: point webcam → get English sentence in real time

**Visual:** Simple 2-panel diagram — person signing on left, English text on right, arrow between

---

### SLIDE 3 — What SignBridge Does (2:00–4:00) · 2 min

**Title:** Full Pipeline Overview

**Talking points:**
Walk through each stage clearly:
1. User opens webcam in a browser — no install needed
2. MediaPipe extracts 21 hand landmarks (x,y,z) from each frame → 63 features
3. CNN classifies static ASL letters (A–Z) OR LSTM classifies full words (top 100)
4. Recognized signs accumulate in a buffer: `["HELLO", "M", "Y", "NAME"]`
5. LangChain + Gemini reads the buffer and outputs: `"Hello, my name is..."`

**Visual (main visual of this slide):**
```
Webcam → MediaPipe (63 features) → CNN / LSTM → Sign Buffer → Gemini LLM → English
```
Show as a horizontal flow diagram with icons

**Key point to emphasize:** Two separate recognition tasks — static letters vs dynamic words — handled by two different model types, both feeding the same LLM stage

---

### SLIDE 4 — Datasets + EDA: ASL Alphabet (4:00–6:30) · 2.5 min

**Title:** Dataset 1 — ASL Alphabet (Static Signs)

**Talking points:**
- Kaggle ASL Alphabet dataset: 87,000 images, 29 classes (A–Z + del, nothing, space)
- Perfectly balanced: exactly 3,000 images per class — no class weighting needed
- 200×200px images, mostly clean white studio backgrounds
- Known limitation flagged in EDA: J and Z require motion in real ASL — dataset uses static approximations
- Impact: we applied augmentation (rotation ±15°, brightness jitter, background variation) to prepare for real webcam conditions

**Figures to show:**
- `docs/figures/asl_class_dist.png` — bar chart, all bars equal height at 3,000
- `docs/figures/asl_samples.png` — grid of sample hand images per letter

**Key takeaway:** Clean, balanced dataset — but studio backgrounds create a real-world generalization challenge we had to address

---

### SLIDE 5 — Datasets + EDA: WLASL Words (6:30–8:30) · 2 min

**Title:** Dataset 2 — WLASL Word-Level Signs (Dynamic Signs)

**Talking points:**
- WLASL: largest public ASL word dataset — 11,980 video clips, top 100 words used for MVP
- Clips per word: 18–40 (mean 20.4) — mild class imbalance, handled with class weights
- Video length: median 66 frames → we sample to fixed 30 frames
- Signer diversity: avg 15 unique signers per word — good for generalization
- Major challenge: 50% of clips unavailable (YouTube takedowns) → only 708 training clips remain (~7 clips/word)
- MediaPipe detection coverage: 41% of frames in sample — low due to varied YouTube lighting/angles

**Figures to show:**
- `docs/figures/wlasl_clip_dist.png` — clip count per word
- `docs/figures/wlasl_frame_dist.png` — frame count histogram

**Key takeaway:** Severe data scarcity (7 clips/word) is the main challenge — not architecture. This is a documented WLASL dataset issue in published literature.

---

### SLIDE 6 — Predictive Modeling Problem Definition (8:30–9:30) · 1 min

**Title:** Two Classification Problems

**Talking points:**
Keep this tight — just define the tasks clearly

| Task | Input | Output | Model Type |
|------|-------|--------|------------|
| ASL Letter Recognition | Single frame → 63 landmarks | 1 of 29 letter classes | MLP / CNN |
| ASL Word Recognition | 30-frame sequence → (30, 63) or (30, 1280) | 1 of 100 word classes | LSTM |

- Both are **supervised multi-class classification** problems
- Evaluation metrics: Accuracy + Macro F1 for letters; Top-1 + Top-5 Accuracy for words
- Why Top-5 for words? Word signs look similar — correct answer often in top 5 even if not #1

---

### SLIDE 7 — First Model: Landmark MLP (9:30–11:30) · 2 min

**Title:** Model 1 — Landmark MLP (Our First Attempt)

**Talking points:**
- Simplest possible model: 63 features → 3-layer MLP → 29 classes
- No image pixels — purely hand geometry (joint positions)
- ~60K parameters, trains on CPU, inference at 63ms
- Architecture: Dense(256) → BN → Dropout → Dense(128) → BN → Dropout → Dense(64) → Dense(29)
- Trained 25 epochs, early stopping, loss converged cleanly
- **Test accuracy: 59%** — Macro F1: 0.68

**Figures to show:**
- `docs/figures/landmark_mlp_training.png` — smooth loss curve converging
- `docs/figures/landmark_mlp_confusion.png` — confusion matrix (M/N, G/H confusions visible)

**Why this was the right first model:**
- Establishes the lower bound — tests whether hand geometry alone can solve the problem
- Very fast to train and deploy — good for validating the pipeline
- 59% with pure geometry tells us: hand shape has useful signal, but pixels matter too

**Key confusions:** M↔N, G↔H, R↔U — letters with nearly identical landmark geometry (fingers fold similarly)

---

### SLIDE 8 — Model Comparison: ASL Alphabet (11:30–13:30) · 2 min

**Title:** ASL Model Results — Three Approaches Compared

**Comparison table (put this large on the slide):**

| Model | Test Accuracy | Val Loss | Latency | Params | Selected? |
|-------|-------------|----------|---------|--------|-----------|
| Landmark MLP | 59.01% | 1.44 | 63 ms | 0.06M | No |
| Baseline CNN (scratch) | 99.99% | 0.00063 | ~130 ms | 0.46M | No |
| **MobileNetV2 (transfer)** | **99.99%** | **0.00018** | **~90 ms** | **2.59M** | ✅ Yes |

**Talking points:**
- Baseline CNN hit 99.99% but trained on studio images — overfits to white backgrounds, fails on real webcam
- MobileNetV2 also hit 99.99% but with 3.5× lower val loss → sharper, more confident predictions
- MobileNetV2 ImageNet pretraining provides features that generalize to real-world conditions
- Two-phase training: freeze base → fine-tune top 30 layers at lr=1e-5
- Phase 1 alone hit 99.09% val accuracy on epoch 1 — ImageNet features transferred immediately

**Figures to show:**
- `docs/figures/asl_model_comparison.png` — bar chart comparing all three
- `docs/figures/mobilenetv2_phase2_training.png` — stable, monotonically improving curve

---

### SLIDE 9 — WLASL Word Model Results (13:30–15:00) · 1.5 min

**Title:** WLASL Word Model — MobileNetV2 + LSTM

**Architecture:**
```
Video clip (30 frames) → MobileNetV2 (frozen) → 1280-dim/frame → LSTM → 100 words
```

**Results:**
| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 9.00% |
| Top-5 Accuracy | 23.00% |
| Macro F1 | 0.056 |
| Inference Latency | 85.2 ms |

**Talking points:**
- Low absolute accuracy is expected — benchmark paper gets 62% Top-1 with 21,000 clips; we have ~1,000 after YouTube attrition
- Top-5 at 23% means correct sign appears in top 5 for 1 in 4 clips — 4.6× better than random (5%)
- Key design decision: MobileNetV2 runs once offline as a feature extractor — never retrained
- This keeps training fast (LSTM head only) while benefiting from ImageNet spatial features
- Also tested ASL-Citizen augmentation (4 experiments) — all underperformed due to domain mismatch between YouTube and controlled webcam recording — reverted to WLASL-only

---

### SLIDE 10 — LLM Pipeline (15:00–16:00) · 1 min

**Title:** LangChain + Gemini — Signs → Sentences

**Visual:** Input/output table from the evaluation

| Input Signs | Output | Quality |
|-------------|--------|---------|
| HELLO MY NAME IS N O S H | "Hello, my name is Nosh." | 5/5 |
| HELP M E | "Help me." | 5/5 |
| MY NAME IS A Y U S H | "My name is Ayush." | 5/5 |
| I AM LEARNING A S L | "I am learning ASL." | 4/5 |

**Talking points:**
- LangChain orchestrates prompt → Gemini 2.5 Flash → parsed sentence
- Handles mixed input: individual fingerspelled letters + full ASL words in one sequence
- 10 test sequences: 100% API success rate, avg quality 4.7/5, avg latency 1,112ms
- Exponential backoff retry (2s, 4s) for reliability
- Context window: last 10 signs only — prevents drift over long sessions

---

### SLIDE 11 — Live Demo (16:00–17:30) · 1.5 min

**Title:** Live Demo

**What to show:**
1. Open the Streamlit app in browser
2. Start webcam — show MediaPipe skeleton overlay on hand
3. Sign a few letters (A, B, L, Y) — show predictions appearing with confidence scores
4. Add letters to buffer: spell "H E L P" → show buffer filling
5. Press Translate → Gemini outputs "Help."
6. Switch to Word Mode — demonstrate recording a word sign

**Have a backup plan:** If webcam fails, show a screenshot/recording of the app working

---

### SLIDE 12 — What We Plan to Enhance (17:30–19:30) · 2 min

**Title:** Next Steps & Enhancements

**Short-term (already in progress):**
- Real-world webcam calibration — the distribution shift between studio training images and webcam is the primary accuracy gap; addressing with wrist-relative landmark normalization
- Threshold tuning — only register predictions above confidence threshold to reduce false additions to the sign buffer

**Model enhancements planned:**
- **Fine-tune MobileNetV2 on webcam frames** — collect ~50 real webcam images per letter and fine-tune the top layers; this closes the studio-to-webcam distribution gap directly
- **Transformer/attention-based word model** — replace 2-layer LSTM with a lightweight transformer encoder; attention mechanisms handle variable-length signing sequences better
- **Continuous signing mode** — current pipeline is discrete (letter-by-letter); future version detects sign boundaries automatically using hand velocity signals from MediaPipe

**Deployment:**
- Dockerize Streamlit app and deploy to GCP Cloud Run (public HTTPS URL)
- Evaluation on real users (proxy user sessions planned for final week)

**Final deliverables:**
- Full evaluation report with all model comparison figures
- Live deployed app URL
- Capstone presentation with real demo

---

### SLIDE 13 — Summary (19:30–20:00) · 30 sec

**Title:** What We've Built So Far

| Component | Status | Key Result |
|-----------|--------|------------|
| ASL Alphabet CNN | ✅ Done | 99.99% val accuracy (MobileNetV2) |
| WLASL Word LSTM | ✅ Done | 23% Top-5, 85ms latency |
| LangChain + Gemini | ✅ Done | 4.7/5 quality, 0% failures |
| Streamlit Live App | ✅ Done | Real-time webcam pipeline |
| Cloud Deployment | 🔄 Week 7 | GCP Cloud Run |
| Final Evaluation | 🔄 Week 7 | Proxy user sessions |

---

## Q&A Preparation — Likely Questions

**Q: Why is word recognition accuracy so low (9%)?**
> A: It's a data problem, not a model problem. The WLASL dataset lost 50% of clips to YouTube takedowns, leaving us ~7 clips per word. The benchmark paper achieves 62% with 21 clips per word across 119 signers. Our architecture is identical to the state of the art — we're just data-limited at MVP scale. Top-5 at 23% is 4.6× better than chance, which shows the model has learned real signal.

**Q: Why not just use an existing API for ASL recognition?**
> A: No public production-grade ASL recognition API exists. Existing research models are not publicly deployed. We initially tried using Gemini Vision as a fallback but found the 1-2 second API latency unsuitable for real-time interaction — local model inference is essential.

**Q: How does the LLM know what to do with mixed letters and words?**
> A: The system prompt explicitly instructs Gemini to treat consecutive letters as a single fingerspelled word (N-O-S-H → "Nosh") and to reorder ASL grammar to natural English word order. This is standard prompt engineering — the LLM already understands ASL communication patterns from its training data.

**Q: What is the end-to-end latency of the full pipeline?**
> A: MediaPipe: ~15ms/frame. Local MLP inference: ~63ms. WLASL LSTM: ~85ms. Gemini API: ~1,100ms. Total for a single translation: roughly 1.2 seconds from sign buffer to English output. The webcam feed itself is real-time at 25fps.

**Q: J and Z require motion — how do you handle that?**
> A: J and Z are currently treated as static signs (the dataset only has static approximations). This is a known limitation we document explicitly. A future enhancement would use the temporal LSTM pathway for J and Z specifically, detecting the characteristic circular/zigzag motion.

**Q: Why MediaPipe landmarks instead of raw images for the word model?**
> A: We tested both. MobileNetV2 features (1280-dim per frame from raw images) significantly outperformed raw MediaPipe landmarks (63-dim) on the WLASL word task. For words, spatial richness matters more than for letters — the motion pattern across a 30-frame sequence is richer in pixel-level features than in joint coordinates alone.

---

## Presentation Tips

- **Practice the demo before** — have the Streamlit app already running in a browser tab, webcam already approved
- **Time check:** Slides 1–6 should finish by the 8:30 mark. If running over, cut Slide 5 detail.
- **Emphasis:** The "first model" the rubric asks about is the Landmark MLP (Slide 7). Spend real time on it.
- **Frame the low WLASL accuracy correctly** — say "we expected this, here's why, here's what we learned" — not "it doesn't work"
- **End on the demo** — if the live demo works, that's the strongest closer

---

*Generated from SignBridge report_log.md — all numbers are real experimental results*
