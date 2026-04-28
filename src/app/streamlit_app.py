"""
SignBridge — Streamlit Web Application (Day 6)
Live pipeline: Webcam → MediaPipe → MobileNetV2 → LangChain/Gemini → English

Video: MJPEG stream via local HTTP server (smooth, no WebRTC, Python 3.13 compatible)

Run:
    streamlit run src/app/streamlit_app.py
"""

import os, sys, io, time, threading, tempfile, concurrent.futures
from collections import deque, Counter
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import cv2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignBridge — ASL Translator",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── MediaPipe Tasks API (Python 3.13 compatible) ─────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

# ─── Constants ────────────────────────────────────────────────────────────────
# Sorted by ASCII on Linux (uppercase A-Z before lowercase del/nothing/space)
# This matches sorted(os.listdir()) used in preprocess_asl.py on the GCP VM
ASL_CLASSES = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "del","nothing","space",
]

BUCKET          = "signbridge-data"
SEQUENCE_LENGTH = 30
MAX_BUFFER      = 10
INFER_EVERY     = 5
MJPEG_PORT      = 8502

# ─── Shared camera state — ONE instance that survives all Streamlit reruns ────
@st.cache_resource
def _get_cam():
    return {
        "running":    False,
        "frame":      None,   # latest annotated BGR np.array
        "result":     None,   # {"sign", "confidence", "top3"}
        "rec_frames": [],
        "recording":  False,
        "mode":       "alphabet",
        "threshold":  0.60,
        "model":      None,
        "lock":       threading.Lock(),
    }

_cam = _get_cam()

# ─── Session state ─────────────────────────────────────────────────────────────
def _init():
    for k, v in {"sign_buffer": [], "history": [], "translation": ""}.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ─── Cached model loaders ─────────────────────────────────────────────────────

@st.cache_resource
def _gemini_vision():
    """Gemini 2.5 Flash vision model — used for ASL letter recognition."""
    import google.generativeai as genai
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    return genai.GenerativeModel("gemini-2.5-flash")


@st.cache_resource(show_spinner="Loading ASL landmark model from GCS…")
def _asl_mlp():
    """Load the ASL Landmark MLP + authoritative class list from GCS."""
    import tensorflow as tf
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(BUCKET)

        data = bucket.blob("models/asl_landmark_mlp_v1.keras").download_as_bytes()
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            f.write(data)
            path = f.name
        model = tf.keras.models.load_model(path)
        os.unlink(path)

        # Load authoritative class order used during training
        try:
            cb = bucket.blob("processed/asl_landmarks/classes.npy").download_as_bytes()
            classes = list(np.load(io.BytesIO(cb), allow_pickle=True))
        except Exception:
            classes = ASL_CLASSES   # fallback to corrected hardcoded list

        return model, classes
    except Exception as e:
        st.warning(f"ASL model load failed: {e}")
        return None, ASL_CLASSES


@st.cache_resource(show_spinner="Loading WLASL model from GCS…")
def _wlasl():
    import tensorflow as tf
    try:
        from google.cloud import storage
        client = storage.Client()
        data = client.bucket(BUCKET).blob(
            "models/wlasl_mobilenetv2_lstm_v1.keras"
        ).download_as_bytes()
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as f:
            f.write(data); path = f.name
        lstm = tf.keras.models.load_model(path)
        os.unlink(path)
        mv2 = tf.keras.applications.MobileNetV2(
            input_shape=(224,224,3), include_top=False,
            pooling="avg", weights="imagenet",
        )
        mv2.trainable = False
        try:
            cb = client.bucket(BUCKET).blob(
                "processed/wlasl_sequences/classes.npy"
            ).download_as_bytes()
            classes = list(np.load(io.BytesIO(cb), allow_pickle=True))
        except Exception:
            classes = [f"WORD_{i}" for i in range(100)]
        return lstm, mv2, classes
    except Exception:
        return None, None, []


@st.cache_resource(show_spinner="Initialising Gemini pipeline…")
def _llm():
    try:
        from src.pipeline.langchain_pipeline import SignBridgePipeline
        return SignBridgePipeline()
    except Exception:
        return None


# ─── MJPEG HTTP server (started once, streams frames to browser) ──────────────

class _MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                with _cam["lock"]:
                    frame = _cam["frame"]
                if frame is not None:
                    ok, jpg = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    if ok:
                        data = jpg.tobytes()
                        self.wfile.write(
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n"
                            + data + b"\r\n"
                        )
                        self.wfile.flush()
                time.sleep(0.04)   # ~25 fps
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, *args):
        pass   # silence HTTP access logs


@st.cache_resource
def _start_mjpeg():
    """Start MJPEG server once — persists across all Streamlit reruns."""
    server = HTTPServer(("0.0.0.0", MJPEG_PORT), _MJPEGHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


_start_mjpeg()   # idempotent — cached, only actually starts once

# ─── Camera + inference background thread ─────────────────────────────────────

def _camera_worker():
    import tensorflow as tf

    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    detector  = mp_vision.HandLandmarker.create_from_options(opts)
    vote_buf  = deque(maxlen=15)   # ~0.5s of frames at 30fps for stability
    frame_n   = 0

    # Load ASL MLP once at thread start
    mlp, asl_classes = _asl_mlp()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        _cam["running"] = False
        return

    while _cam["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_n += 1

        # MediaPipe landmark detection + skeleton overlay
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det    = detector.detect(mp_img)

        annotated    = frame.copy()
        hand_visible = bool(det.hand_landmarks)

        if hand_visible:
            hand = det.hand_landmarks[0]
            h, w = frame.shape[:2]
            pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for c in HAND_CONNECTIONS:
                cv2.line(annotated, pts[c[0]], pts[c[1]], (0, 200, 0), 2)
            for pt in pts:
                cv2.circle(annotated, pt, 5, (0, 255, 80), -1)

        with _cam["lock"]:
            mode      = _cam["mode"]
            threshold = _cam["threshold"]
            recording = _cam["recording"]

        # ── Local MLP inference — runs every frame, instant, no API lag ───────
        if mode == "alphabet" and hand_visible and mlp is not None:
            hand = det.hand_landmarks[0]
            lm = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)  # (21,3)

            # Wrist-relative normalization using palm length as stable scale reference
            # Landmark 9 = middle finger MCP (knuckle) — fixed distance from wrist
            # regardless of hand pose, unlike max-distance which varies per sign
            lm -= lm[0:1]                                    # center on wrist
            scale = np.linalg.norm(lm[9])                   # palm length
            if scale > 1e-6:
                lm /= scale

            lm_vec = lm.flatten()[np.newaxis]   # (1, 63)

            probs   = mlp(lm_vec, training=False).numpy()[0]
            top_idx = int(np.argmax(probs))
            conf    = float(probs[top_idx])
            sign    = asl_classes[top_idx] if top_idx < len(asl_classes) else f"C{top_idx}"

            vote_buf.append(sign)
            voted     = Counter(vote_buf).most_common(1)[0][0]
            vote_conf = Counter(vote_buf)[voted] / len(vote_buf)

            # Top-3 from this frame's probs
            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = [(asl_classes[i] if i < len(asl_classes) else f"C{i}", float(probs[i]))
                    for i in top3_idx]

            with _cam["lock"]:
                _cam["result"] = {
                    "sign":       voted,
                    "confidence": vote_conf,
                    "raw_conf":   conf,
                    "top3":       top3,
                }
        elif mode == "alphabet" and not hand_visible:
            vote_buf.clear()

        # Accumulate frames for word mode
        if mode == "word" and recording:
            with _cam["lock"]:
                _cam["rec_frames"].append(frame.copy())

        # Overlay prediction on the frame
        with _cam["lock"]:
            res = _cam["result"]

        if mode == "alphabet" and res:
            sign, conf = res["sign"], res["confidence"]
            color = (0, 220, 0) if conf >= threshold else (0, 140, 255)
            label = f"{sign}  {conf:.0%}"
            cv2.putText(annotated, label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
            cv2.putText(annotated, label, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)

        if mode == "word":
            with _cam["lock"]:
                n = len(_cam["rec_frames"])
            status = f"REC {n}/{SEQUENCE_LENGTH}" if recording else "WORD MODE"
            cv2.putText(annotated, status, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 200), 3)

        with _cam["lock"]:
            _cam["frame"] = annotated

        time.sleep(0.01)

    cap.release()
    detector.close()


def start_camera():
    if _cam["running"]:
        return
    _cam["result"]  = None
    _cam["frame"]   = None
    _cam["running"] = True
    threading.Thread(target=_camera_worker, daemon=True).start()


def stop_camera():
    _cam["running"] = False


# ─── WLASL word inference ─────────────────────────────────────────────────────

def predict_word(frames_bgr, threshold):
    import tensorflow as tf
    lstm, mv2, classes = _wlasl()
    if lstm is None or not frames_bgr:
        return None, 0.0
    feats = []
    for f in frames_bgr:
        rgb = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224))
        x   = tf.keras.applications.mobilenet_v2.preprocess_input(rgb.astype(np.float32))
        feats.append(mv2(x[np.newaxis], training=False).numpy()[0])
    feats = np.array(feats, dtype=np.float32)
    if len(feats) >= SEQUENCE_LENGTH:
        idx   = np.linspace(0, len(feats)-1, SEQUENCE_LENGTH, dtype=int)
        feats = feats[idx]
    else:
        feats = np.vstack([feats, np.zeros((SEQUENCE_LENGTH-len(feats), 1280), np.float32)])
    probs = lstm.predict(feats[np.newaxis], verbose=0)[0]
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    word  = classes[idx] if idx < len(classes) else f"CLASS_{idx}"
    return (word, conf) if conf >= threshold else (None, conf)


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("SignBridge — ASL to English Translator")
st.caption("Live webcam · MediaPipe landmarks · MobileNetV2 · Gemini translation")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    mode_label = st.radio("Mode", ["Alphabet (letters)", "Words (WLASL)"])
    mode_key   = "alphabet" if "Alphabet" in mode_label else "word"
    threshold  = st.slider("Confidence threshold", 0.30, 1.0, 0.60, 0.05)

    with _cam["lock"]:
        _cam["mode"]      = mode_key
        _cam["threshold"] = threshold

    st.divider()
    cc1, cc2 = st.columns(2)
    if cc1.button("▶ Start", use_container_width=True):
        start_camera()
    if cc2.button("⏹ Stop", use_container_width=True):
        stop_camera()

    st.divider()
    st.subheader("📋 Sign Buffer")
    buf_slot = st.empty()

    bc1, bc2 = st.columns(2)
    add_btn   = bc1.button("➕ Add Sign", use_container_width=True)
    clear_btn = bc2.button("🗑️ Clear",   use_container_width=True)

    st.divider()
    translate_btn = st.button(
        "🌐 Translate (LLM)", use_container_width=True, type="primary"
    )

# ── Main columns ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("📷 Live Feed")

    if _cam["running"]:
        # MJPEG stream — smooth, no Streamlit DOM involvement
        st.components.v1.html(
            f"""
            <img src="http://localhost:{MJPEG_PORT}"
                 style="width:100%;border-radius:8px;display:block;"
                 title="Live ASL feed" />
            """,
            height=420,
            scrolling=False,
        )
    else:
        st.info("Press **▶ Start** in the sidebar to open your webcam.")

    # Word-mode controls
    if mode_key == "word" and _cam["running"]:
        st.divider()
        wc1, wc2 = st.columns(2)
        if wc1.button("⏺ Record Word", use_container_width=True):
            with _cam["lock"]:
                _cam["rec_frames"] = []
                _cam["recording"]  = True
            st.info(f"Recording — hold the sign for ~{SEQUENCE_LENGTH} frames.")
        if wc2.button("⏹ Stop & Predict", use_container_width=True):
            with _cam["lock"]:
                _cam["recording"] = False
                frames = list(_cam["rec_frames"])
                _cam["rec_frames"] = []
            if frames:
                with st.spinner(f"Running WLASL LSTM on {len(frames)} frames…"):
                    word, conf = predict_word(frames, threshold)
                if word:
                    st.success(f"Predicted: **{word}** ({conf:.1%})")
                    if len(st.session_state.sign_buffer) >= MAX_BUFFER:
                        st.session_state.sign_buffer.pop(0)
                    st.session_state.sign_buffer.append(word)
                else:
                    st.warning(f"Low confidence ({conf:.1%}) — try again.")
            else:
                st.warning("No frames — press Record first.")

with right_col:
    st.subheader("🔤 Prediction")
    with _cam["lock"]:
        res = _cam["result"]

    if res and mode_key == "alphabet":
        sign, conf = res["sign"], res["confidence"]
        color = "green" if conf >= threshold else "orange"
        alts  = ""
        if res.get("top3"):
            alt_list = [f"{s} {c:.0%}" for s, c in res["top3"][1:]]
            alts = f"<p style='text-align:center;color:gray;font-size:0.9em'>Also: {' · '.join(alt_list)}</p>"
        st.markdown(
            f"<h1 style='color:{color};text-align:center'>{sign}</h1>"
            f"<p style='text-align:center'>Confidence: <b>{conf:.1%}</b></p>" + alts,
            unsafe_allow_html=True,
        )
        st.caption("Prediction refreshes when you interact with the page. The video overlay updates live.")
    elif mode_key == "word":
        st.markdown(
            "<p style='text-align:center'>Press <b>Record Word</b> · sign · <b>Stop & Predict</b></p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:gray;text-align:center'>Start camera and show your hand</p>",
            unsafe_allow_html=True,
        )

    st.subheader("💬 Translation")
    trans_slot = st.empty()
    if st.session_state.translation:
        trans_slot.success(st.session_state.translation)
    else:
        trans_slot.info("Build a sign buffer then press **Translate**")

# ── Button handlers ────────────────────────────────────────────────────────────
if add_btn:
    with _cam["lock"]:
        res = _cam["result"]
    if res and mode_key == "alphabet":
        sign, conf = res["sign"], res["confidence"]
        if conf >= threshold and sign not in ("nothing",):
            tok = " " if sign == "space" else sign
            if len(st.session_state.sign_buffer) >= MAX_BUFFER:
                st.session_state.sign_buffer.pop(0)
            st.session_state.sign_buffer.append(tok)
    else:
        st.sidebar.warning("No prediction — start camera first.")

if clear_btn:
    st.session_state.sign_buffer = []
    st.session_state.translation = ""

if translate_btn:
    llm = _llm()
    buf = [s for s in st.session_state.sign_buffer if s.strip()]
    if not buf:
        trans_slot.warning("Buffer is empty.")
    elif llm is None:
        trans_slot.error("LLM unavailable — check GOOGLE_API_KEY in .env")
    else:
        with st.spinner("Calling Gemini…"):
            result = llm.translate(buf)
        sentence = result["sentence"]
        st.session_state.translation = sentence
        st.session_state.history.append({
            "buffer": " ".join(buf),
            "sentence": sentence,
            "latency_ms": result["latency_ms"],
        })
        trans_slot.success(sentence)

# ── Buffer display ─────────────────────────────────────────────────────────────
buf_slot.markdown(
    " → ".join(st.session_state.sign_buffer)
    if st.session_state.sign_buffer else "_empty_"
)

# ── Session history ────────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("📜 Session History")
    for i, h in enumerate(reversed(st.session_state.history), 1):
        st.markdown(
            f"**{i}.** `{h['buffer']}` → _{h['sentence']}_ "
            f"<span style='color:gray;font-size:0.85em'>({h['latency_ms']:.0f} ms)</span>",
            unsafe_allow_html=True,
        )
