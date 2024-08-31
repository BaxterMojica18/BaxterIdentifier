import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch
import numpy as np

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

# Load the model on app startup (assuming it's lightweight)
model = YOLO('best.pt').to(get_device())

def predict(image):
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)

    if len(detections) == 0:
        return None

    for detection in detections:
        box, score, class_id = detection[:3]
        class_id = int(class_id)
        detected_name = model.names[class_id]
        if detected_name == "Baxter":
            return "A Baxter is detected!"

    return None

def initialize_window_and_capture(window_width, window_height):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    cap.set(3, window_width)
    cap.set(4, window_height)
    cap.set(cv2.CAP_PROP_EXPOSURE, -3)
    frame_rate = 30
    cap.set(cv2.CAP_PROP_FPS, frame_rate)

    return cap

st.title("Baxter Detection")

source_selection = st.radio("Choose source:", ("Image Upload", "Live Camera"))

if source_selection == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")

        if st.button("Predict", key="predict_image"):
            detected_name = predict(image)
            if detected_name:
                st.success("Baxter detected in the image!")
            else:
                st.warning("No Baxter detected in the image.")

elif source_selection == "Live Camera":
    frame_width = 640
    frame_height = 480

    if "cap" not in st.session_state:
        st.session_state.cap = initialize_window_and_capture(frame_width, frame_height)

    if "run" not in st.session_state:
        st.session_state.run = False

    start_button = st.button("Connect Model", key="start_button")
    
    if start_button:
      st.success(f"Model Connected")

    if start_button or st.session_state.run:
        st.session_state.run = True
        cap = st.session_state.cap

        frame_placeholder = st.empty()
        
         # Camera input
    frame = st.camera_input(label_visibility="hidden", label="hello", key="inputcamera2")

    # If the camera feed is running
    if frame and st.session_state.run:
        # Decode the image
        image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the original image
        #st.image(image, channels="BGR")

        # Perform prediction
        detected_name = predict(rgb_frame)
        if detected_name:
            st.success(f"Detected object: {detected_name}")
        else:
            st.warning("No Baxter detected.")


   
