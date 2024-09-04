import streamlit as st
import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch
import pandas as pd
import datetime
import os

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

def save_detection(detection_number, detection_name, date_of_detection, csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    data = {
        "Detection Number": [detection_number],
        "Detection Name": [detection_name],
        "Date of Detection": [date_of_detection]
    }
    df = pd.DataFrame(data)
    
    # Append to the CSV file
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

st.title("Baxter Detection")

source_selection = st.radio("Choose source:", ("Image Upload", "Live Camera"))

if source_selection == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])
    frame_placeholder = st.empty()
    detection_count = 0

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")

        if st.button("Predict", key="predict_image"):
            detected_name = predict(image)
            if detected_name:
                st.success("Baxter detected in the image!")
                 # Save detection to CSV
                detection_count += 1
                date_of_detection = datetime.datetime.now().strftime("%Y-%m-%d")
                save_detection(detection_count, detected_name, date_of_detection)
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
        detection_count = 0

        while st.session_state.run:
            frame = st.camera_input(label_visibility="hidden", label="hello", key="inputcamera2")
            
            if frame:
                # Decode the image
                image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Perform prediction
                detected_name = predict(rgb_frame)
                if detected_name:
                    st.success(f"Detected object: {detected_name}")

                    # Save detection to CSV
                    detection_count += 1
                    date_of_detection = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_detection(detection_count, detected_name, date_of_detection)
                else:
                    st.warning("No Baxter detected.")

            # Button to stop the camera
            if st.button("Stop Camera", key="stop_camera"):
                st.session_state.run = False
                cap.release()
                cv2.destroyAllWindows()
                st.stop()  # This stops the execution and removes the camera feed from the UI
