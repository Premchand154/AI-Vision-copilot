import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import sys
import os
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from captioning.blip_caption import generate_caption
from reasoning.llm_reasoning import ask_llm
from realtime_detection import detect_objects

st.set_page_config(page_title="AI Vision Copilot", layout="wide")
st.title("AI Vision Copilot")

@st.cache_resource
def load_models():
    return True

load_models()

mode = st.selectbox("Mode", ["Upload Image", "Live Camera"])

question = st.text_input("Ask a question about the scene:")
ask_button = st.button("Ask")

# ---------------- IMAGE MODE ----------------
if mode == "Upload Image":

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)

        st.image(image, caption="Uploaded Image", use_container_width=True)

        objects = detect_objects(frame)
        caption = generate_caption(frame)

        st.write("Objects:", objects)
        st.write("Caption:", caption)

        if ask_button:
            if not question:
                st.warning("Please enter a question")
            else:
                answer = ask_llm(caption, objects, question)
                st.write("Answer:")
                st.write(answer)

# ---------------- LIVE CAMERA MODE ----------------
elif mode == "Live Camera":

    class VisionProcessor(VideoTransformerBase):
        def __init__(self):
            self.caption = ""
            self.objects = []
            self.frame_count = 0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            self.objects = detect_objects(img)

            if self.frame_count % 30 == 0:
                self.caption = generate_caption(img)

            self.frame_count += 1

            for i, obj in enumerate(self.objects):
                cv2.putText(img, obj, (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            return img

    ctx = webrtc_streamer(
        key="vision",
        video_processor_factory=VisionProcessor
    )

    if ctx.video_processor:
        processor = ctx.video_processor

        if ask_button:
            if not processor.caption:
                st.warning("Waiting for caption...")
            elif not question:
                st.warning("Please enter a question")
            else:
                answer = ask_llm(processor.caption, processor.objects, question)

                st.write("Answer:")
                st.write(answer)