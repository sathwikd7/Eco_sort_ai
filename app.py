import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from gtts import gTTS
import base64
import io
from collections import Counter

st.set_page_config(page_title="EcoSort AI", layout="wide")

# ✅ Your exact model filename
MODEL_PATH = "ecosort_ai_model.keras"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except:
        return tf.keras.models.load_model(MODEL_PATH, safe_mode=False, compile=False)
model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

def preprocess_image(img):
    img = img.resize((96, 96))
    arr = np.array(img).astype("float32") / 255.0
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(img):
    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    label = CLASS_NAMES[idx]
    st.session_state.history.append({"label": label, "confidence": conf})
    return label, conf, preds

def speak_text(text):
    tts = gTTS(text=text, lang="en")
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    b64 = base64.b64encode(fp.read()).decode()
    st.markdown(f'<audio controls autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

st.sidebar.title("🗑️ EcoSort AI")
page = st.sidebar.radio("Navigation", ["📁 Upload", "📷 Camera", "🔊 Voice", "📊 Analytics"])

st.title("🗑️ EcoSort AI")
st.markdown("**Waste classification with upload, camera, voice assistant, and analytics dashboard.**")

if page == "📁 Upload":
    st.header("📁 Upload Prediction")
    uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            label, conf, preds = predict_image(img)
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {conf:.2f}")
            st.bar_chart(pd.DataFrame({"Class": CLASS_NAMES, "Prob": preds}).set_index("Class"))
            if st.button("🔊 Speak Result"):
                speak_text(f"The predicted waste category is {label} with {conf:.2f} confidence")

elif page == "📷 Camera":
    st.header("📷 Live Camera")
    cam = st.camera_input("Take a picture")
    if cam is not None:
        img = Image.open(cam).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Camera Image", use_container_width=True)
        with col2:
            label, conf, preds = predict_image(img)
            st.success(f"**Prediction:** {label}")
            st.info(f"**Confidence:** {conf:.2f}")
            if st.button("🔊 Speak Camera Result"):
                speak_text(f"Camera detected {label} with {conf:.2f} confidence")

elif page == "🔊 Voice":
    st.header("🔊 Voice Assistant")
    text = st.text_input("Text to speak", value="Welcome to EcoSort AI")
    if st.button("Generate Voice"):
        speak_text(text)

elif page == "📊 Analytics":
    st.header("📊 Analytics Dashboard")
    if not st.session_state.history:
        st.warning("Make some predictions first!")
    else:
        df = pd.DataFrame(st.session_state.history)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Class Distribution")
            counts = df["label"].value_counts()
            fig, ax = plt.subplots()
            ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            st.pyplot(fig)
        with col2:
            st.subheader("Confidence Trend")
            st.line_chart(df["confidence"])
        st.subheader("Recent Predictions")
        st.dataframe(df.tail(10))