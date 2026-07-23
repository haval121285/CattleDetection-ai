from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
ASSETS = ROOT / "assets"

st.set_page_config(page_title="CattleVision-AI", page_icon="🐄", layout="wide")

st.markdown(
    """
<style>
:root { --ink:#10221c; --green:#1f7a5a; --mint:#dff5ea; --gold:#e9b949; }
.stApp { background:linear-gradient(135deg,#f7fbf8 0%,#edf7f1 55%,#fffaf0 100%); }
.block-container { max-width:1250px; padding-top:1.5rem; }
.hero { padding:2rem 2.2rem; border-radius:24px; color:white;
  background:linear-gradient(120deg,#102f25,#1f7a5a); box-shadow:0 16px 45px #153b2d26; }
.hero h1 { margin:0; font-size:clamp(2rem,5vw,4rem); letter-spacing:-2px; }
.hero p { color:#d9f3e7; font-size:1.08rem; margin:.5rem 0 0; }
.pill { display:inline-block; padding:.35rem .75rem; margin:.25rem .3rem .25rem 0;
  border:1px solid #ffffff55; border-radius:99px; font-size:.82rem; }
.metric-card { background:#ffffffcc; border:1px solid #d8e8df; border-radius:18px;
  padding:1rem; box-shadow:0 8px 24px #153b2d12; min-height:118px; }
.metric-card .v { color:#1f7a5a; font-size:1.75rem; font-weight:800; }
.metric-card .k { color:#567067; font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }
.result { text-align:center; padding:1.5rem; border-radius:22px; background:#e5f7ed;
  border:1px solid #acd8bf; }
.weight { font-size:3.2rem; color:#146747; font-weight:850; line-height:1; }
.soft { color:#5c7169; }
[data-testid="stFileUploader"] { background:#ffffffaa; padding:1rem; border-radius:18px; }
</style>
""",
    unsafe_allow_html=True,
)


def read_json(name: str) -> dict:
    with (MODELS / name).open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource(show_spinner="Loading AI models…")
def load_resources():
    import tensorflow as tf
    from ultralytics import YOLO

    deployment = read_json("deployment_config.json")
    gate = read_json("cow_gate_config.json")
    stats = read_json("model_stats.json")
    detector = YOLO(str(MODELS / gate["detector"]))
    runner = {"mode": "tflite", "keras": None, "embedding": None, "references": None}

    # Prefer full Keras because it also exposes the 128-D OOD embedding. The
    # exported TFLite model is a reliable fallback when a copied .keras archive
    # is incomplete or incompatible with the local TensorFlow version.
    try:
        model = tf.keras.models.load_model(
            MODELS / deployment["regression_model_file"], compile=False
        )
        runner["mode"] = "keras"
        runner["keras"] = model
        runner["embedding"] = tf.keras.Model(
            inputs=model.input, outputs=model.get_layer("embedding").output
        )
        references = np.load(MODELS / "cow_gate_reference_embeddings.npz")["embeddings"]
        references = references.astype("float32")
        references /= np.maximum(np.linalg.norm(references, axis=1, keepdims=True), 1e-8)
        runner["references"] = references
    except Exception:
        interpreter = tf.lite.Interpreter(
            model_path=str(MODELS / "cattleweight_model.tflite")
        )
        interpreter.allocate_tensors()
        runner["interpreter"] = interpreter
        runner["input"] = interpreter.get_input_details()[0]
        runner["output"] = interpreter.get_output_details()[0]

    return runner, detector, deployment, gate, stats


def run_regression(runner: dict, tensor: np.ndarray) -> float:
    if runner["mode"] == "keras":
        return float(np.asarray(runner["keras"].predict(tensor, verbose=0)).squeeze())

    interpreter = runner["interpreter"]
    input_info = runner["input"]
    model_input = tensor.astype(input_info["dtype"], copy=False)
    interpreter.set_tensor(input_info["index"], model_input)
    interpreter.invoke()
    return float(np.asarray(interpreter.get_tensor(runner["output"]["index"])).squeeze())


def to_tensor(image: Image.Image, size: int) -> np.ndarray:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, ...]


def analyze(image: Image.Image, resources: tuple) -> dict:
    runner, detector, deployment, gate, stats = resources
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    detection = detector.predict(
        source=arr,
        classes=[int(gate["cow_class_id"])],
        conf=float(gate["cow_confidence"]),
        iou=0.50,
        max_det=5,
        verbose=False,
    )[0]

    width, height = rgb.size
    image_area = max(width * height, 1)
    valid = []
    if detection.boxes is not None:
        for box, conf in zip(
            detection.boxes.xyxy.cpu().numpy(), detection.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(float, box)
            ratio = max(0, x2 - x1) * max(0, y2 - y1) / image_area
            if ratio >= float(gate["min_cow_area_ratio"]):
                valid.append((x1, y1, x2, y2, float(conf), float(ratio)))

    if not valid:
        return {"ok": False, "title": "No suitable cow detected", "message":
                "Use a clear image where one cow is large and fully visible."}
    if len(valid) > int(gate["max_cows_allowed"]):
        return {"ok": False, "title": "Multiple cows detected", "message":
                f"Detected {len(valid)} cows. Please upload an image containing only one cow."}

    x1, y1, x2, y2, confidence, area_ratio = max(valid, key=lambda x: x[4] * x[5])
    padding = float(gate["crop_padding"])
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    crop_box = (
        max(0, int(x1 - px)), max(0, int(y1 - py)),
        min(width, int(x2 + px)), min(height, int(y2 + py)),
    )
    crop = rgb.crop(crop_box)
    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, width // 220)
    draw.rectangle(crop_box, outline="#20e28a", width=line_width)

    tensor = to_tensor(crop, int(gate["image_size"]))
    similarity = None
    if runner["mode"] == "keras":
        embedding = runner["embedding"].predict(tensor, verbose=0).reshape(-1).astype("float32")
        embedding /= max(float(np.linalg.norm(embedding)), 1e-8)
        similarity = float(np.max(runner["references"] @ embedding))
        if similarity < float(gate["similarity_threshold"]):
            return {"ok": False, "title": "Unfamiliar cattle image", "message":
                    "A cow was detected, but its appearance is too different from the training data. Try another clear side or rear image.",
                    "annotated": annotated, "crop": crop, "confidence": confidence,
                    "similarity": similarity}

    normalized_weight = run_regression(runner, tensor)
    weight = normalized_weight * float(deployment["weight_std"]) + float(deployment["weight_mean"])
    if not float(gate["safe_min_weight"]) <= weight <= float(gate["safe_max_weight"]):
        return {"ok": False, "title": "Prediction outside safe range", "message":
                "The model produced an unusual value, so the estimate was withheld.",
                "annotated": annotated, "crop": crop}

    return {
        "ok": True, "weight": weight, "mae": float(stats["mae_kg"]),
        "confidence": confidence, "similarity": similarity, "model_mode": runner["mode"],
        "area_ratio": area_ratio, "annotated": annotated, "crop": crop,
    }


st.markdown(
    """<section class="hero"><span class="pill">FINAL PROJECT</span>
    <span class="pill">COMPUTER VISION</span><span class="pill">LOCAL & PRIVATE</span>
    <h1>CattleVision-AI</h1><p>Contactless cattle weight estimation from a single image</p></section>""",
    unsafe_allow_html=True,
)

try:
    stats = read_json("model_stats.json")
except Exception as error:
    st.error(f"Configuration error: {error}")
    st.stop()

st.write("")
cards = [
    ("Test MAE", f"{stats['mae_kg']:.2f} kg"),
    ("Test RMSE", f"{stats['rmse_kg']:.2f} kg"),
    ("Test images", f"{stats['test_images']:,}"),
    ("Unseen test cattle", f"{stats['test_animals']:,}"),
]
columns = st.columns(4)
for column, (label, value) in zip(columns, cards):
    column.markdown(f'<div class="metric-card"><div class="k">{label}</div><div class="v">{value}</div></div>', unsafe_allow_html=True)

predict_tab, performance_tab, about_tab = st.tabs(["🔍 Estimate Weight", "📊 Model Performance", "ℹ️ About"])

with predict_tab:
    st.subheader("Upload one clear cattle image")
    uploaded = st.file_uploader("JPG, JPEG or PNG", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            source_image = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("This file is not a valid image.")
            st.stop()
        st.image(source_image, caption="Uploaded image", width="stretch")
        if st.button("Analyze cattle image", type="primary", use_container_width=True):
            try:
                resources = load_resources()
                with st.spinner("Detecting the cow and estimating weight…"):
                    result = analyze(source_image, resources)
                st.session_state["last_result"] = result
            except ModuleNotFoundError as error:
                st.error(f"A required package is missing: {error.name}. Run: pip install -r requirements.txt")
            except Exception as error:
                st.error(f"Analysis could not finish: {error}")

    result = st.session_state.get("last_result")
    if uploaded and result:
        if not result["ok"]:
            st.warning(f"**{result['title']}** — {result['message']}")
            if "annotated" in result:
                left, right = st.columns(2)
                left.image(result["annotated"], caption="Detection", width="stretch")
                right.image(result["crop"], caption="Model crop", width="stretch")
        else:
            low = max(0.0, result["weight"] - result["mae"])
            high = result["weight"] + result["mae"]
            st.markdown(
                f'<div class="result"><div class="soft">ESTIMATED LIVE WEIGHT</div>'
                f'<div class="weight">{result["weight"]:.1f} kg</div>'
                f'<div class="soft">Typical error-based range: {low:.1f}–{high:.1f} kg</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            left, right = st.columns(2)
            left.image(result["annotated"], caption=f"Cow detection confidence: {result['confidence']:.1%}", width="stretch")
            crop_caption = "Validated crop"
            if result["similarity"] is not None:
                crop_caption += f" · similarity: {result['similarity']:.3f}"
            else:
                crop_caption += " · TFLite prediction mode"
            right.image(result["crop"], caption=crop_caption, width="stretch")
            st.info("This is an AI estimate for educational use. Use a calibrated livestock scale for veterinary, sale, dosage, or feeding decisions.")

with performance_tab:
    st.subheader("Evaluation on unseen cattle")
    plot_path = ASSETS / "evaluation_plots.png"
    if plot_path.exists():
        st.image(str(plot_path), caption="Actual vs predicted weights and absolute-error distribution", width="stretch")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAPE", f"{stats['mape_percent']:.2f}%")
    col2.metric("Within ±20 kg", f"{stats['within_20kg_percent']:.2f}%")
    col3.metric("R²", f"{stats['r2']:.3f}")
    st.caption("The split used GroupShuffleSplit by animal ID (70/15/15), preventing images of the same animal from appearing in both training and evaluation sets.")

with about_tab:
    st.subheader("How the system works")
    st.markdown("""
1. **YOLO11s detection** confirms that exactly one sufficiently large cow is visible.
2. **Automatic cropping** isolates the detected animal from the background.
3. **Embedding similarity gate** rejects unfamiliar cattle images when the full Keras archive is available.
4. **EfficientNetV2B0 regression** uses Keras or its TFLite export to estimate normalized weight and convert it to kilograms.

The model was trained on **13,468 images from 1,071 cattle** and evaluated on **2,928 images from 230 unseen cattle**. All inference runs locally on your computer; uploaded images are not sent to a cloud service.
""")
    st.warning("Current model limitation: R² = 0.215 and MAE = 25.68 kg. Lighting, camera angle, distance, occlusion, breed, and body condition can affect accuracy.")

st.markdown("<p style='text-align:center;color:#698078;margin-top:2rem'>CattleVision-AI · Final Project · Summer 2026</p>", unsafe_allow_html=True)
