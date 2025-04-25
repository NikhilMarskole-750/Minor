import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# Load YOLO model
model = YOLO("indian_road.pt")
alert_classes = ["barricade", "person", "bike", "rickshaw", "vehicle", "animal", "pole", "thela", "traffic signal", "obstruction", "patchhole"]

st.title("🚗 Intelligent Vehicle Assistance System")
mode = st.radio("Select Mode", ("Video File", "Webcam"))

# Process a single frame and draw alerts
def process_frame(frame):
    results = model(frame)[0]
    alert_texts = []

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)
        class_name = model.names[class_id]
        box_width = x2 - x1

        if class_name in alert_classes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if box_width > 250:
                alert_texts.append(f"🚨 IMPACT ALERT: {class_name} extremely close!")
            elif box_width > 150:
                alert_texts.append(f"⚠️ WARNING: {class_name} close")
            elif box_width > 80:
                alert_texts.append(f"🟡 CAUTION: {class_name} ahead")

    return frame, alert_texts

# Define webrtc video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.alert_placeholder = st.empty()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_frame, alerts = process_frame(img)

        # Show alerts
        with self.alert_placeholder:
            if alerts:
                st.warning("\n".join(alerts))
            else:
                st.info("✅ No immediate danger")

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")

# Main logic
if mode == "Video File":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        alert_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, alerts = process_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

            if alerts:
                alert_placeholder.warning("\n".join(alerts))
            else:
                alert_placeholder.info("✅ No immediate danger")

        cap.release()

elif mode == "Webcam":
    st.info("🔴 Please allow webcam access when prompted.")
    webrtc_streamer(key="live", video_processor_factory=VideoProcessor)
