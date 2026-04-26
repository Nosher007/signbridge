# SignBridge

**AI-Powered Sign Language Recognition and Context-Aware Translation System**

SignBridge is a full-stack AI application that translates American Sign Language (ASL) into natural English sentences in real time using computer vision, deep learning, and large language models.

---

## Stack

- **Hand Detection:** MediaPipe Hands (21 landmarks)
- **Static Sign Model:** MobileNetV2 (ASL Alphabet — 29 classes)
- **Dynamic Sign Model:** MobileNetV2 + LSTM (WLASL Top-100 words)
- **LLM Pipeline:** LangChain + Gemini 1.5 Flash via GCP Vertex AI
- **Web UI:** Streamlit + streamlit-webrtc
- **Infra:** GCP Cloud Run + GCS + Vertex AI

---

## Repo Structure

```
signbridge/
├── data/               # Local dataset cache (not committed — use GCS)
├── notebooks/          # EDA notebooks
├── src/
│   ├── data/           # Preprocessing scripts
│   ├── models/         # CNN and LSTM model definitions
│   ├── pipeline/       # MediaPipe extractor + LangChain pipeline
│   └── app/            # Streamlit web app
├── docs/
│   ├── report_log.md   # Living report (filled daily)
│   └── figures/        # All plots and charts
├── tests/
│   └── test_log.md     # Daily test results
├── configs/
│   └── config.yaml     # Paths, hyperparameters, flags
├── requirements.txt
└── Dockerfile
```

---

## Setup

```bash
git clone https://github.com/Nosher007/signbridge.git
cd signbridge
pip install -r requirements.txt
```

> Datasets and model checkpoints are stored in GCS (`gs://signbridge-data/`), not in this repo.

---

## Team

- **Nosherwan Babar** — Data pipeline, LLM integration, UI, Cloud Run deployment
- **Ayush Gundawar** — MediaPipe extractor, CNN model, LSTM model, evaluation
