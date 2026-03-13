import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import av
import cv2

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# -----------------------
# Config
# -----------------------
IMG_SIZE = (128, 128)  # must match Colab training [page:1]


# -----------------------
# Load model + class names
# -----------------------
@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model("model/rps_model.h5")
    with open("model/class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names


model, class_names = load_model_and_classes()
st.sidebar.write("Loaded classes:", class_names)  # debug: should be ['rock', 'paper', 'scissors'] [page:1]


# -----------------------
# Preprocess webcam frame
# -----------------------
def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    # frame_bgr: OpenCV-style BGR image from webcam
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)  # (128,128) [page:1]
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1,128,128,3)
    return img


# -----------------------
# Video transformer
# -----------------------
class RPSVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.label_text = "Show rock, paper, or scissors ✊🖐️✌️"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Flip horizontally to act like a mirror
        img = cv2.flip(img, 1)

        try:
            input_tensor = preprocess_frame(img)
            preds = model.predict(input_tensor, verbose=0)

            # Debug: show raw predictions in browser sidebar
            st.sidebar.write("Preds:", preds)

            prob = float(np.max(preds))
            cls_idx = int(np.argmax(preds))
            pred_class = class_names[cls_idx]

            self.label_text = f"{pred_class.upper()} ({prob*100:.1f}%)"
        except Exception as e:
            self.label_text = f"Error: {e}"

        # Draw label banner on the frame
        cv2.rectangle(img, (10, 10), (520, 70), (0, 0, 0), -1)
        cv2.putText(
            img,
            self.label_text,
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------
# Streamlit UI
# -----------------------
st.title("Rock–Paper–Scissors Live Classifier 🎥✊🖐️✌️")
st.write(
    "Allow camera access, then show **rock**, **paper**, or **scissors** with your hand. "
    "The model will try to guess your move in real time."
)

webrtc_streamer(
    key="rps",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=RPSVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
