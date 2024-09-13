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

def get_current_detection_count(csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    """Get the current detection count from the CSV file."""
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip')  # Skip bad lines
            if len(df) > 0:
                return df["Detection Number"].max()
        except pd.errors.EmptyDataError:
            return 0  # Handle case where file exists but is empty
    return 0

def correct_detection_numbers(csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    """Check and correct duplicate detection numbers in the CSV file."""
    if os.path.exists(csv_file):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Sort by detection number and reset the index
            df = df.sort_values(by="Detection Number").reset_index(drop=True)

            # Correct duplicate detection numbers
            last_detection_number = 0
            for idx in range(len(df)):
                current_number = df.loc[idx, "Detection Number"]
                # If current detection number is less than or equal to the last, fix it
                if current_number <= last_detection_number:
                    df.loc[idx, "Detection Number"] = last_detection_number + 1
                last_detection_number = df.loc[idx, "Detection Number"]

            # Save the updated DataFrame back to the CSV file
            df.to_csv(csv_file, index=False)
            st.success("Detection numbers have been corrected.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.warning("CSV file does not exist.")
        

def save_detection(detection_number, detection_name, date_of_detection, time_of_detection, csv_file="E:/Downloads/ImageIdentification/detections.csv"):
    data = {
        "Detection Number": [detection_number],
        "Detection Name": [detection_name],
        "Date of Detection": [date_of_detection],
        "Time of Detection": [time_of_detection]
    }
    df = pd.DataFrame(data)
    
    # Append to the CSV file
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

# Initialize detection count from existing CSV file
if "detection_count" not in st.session_state:
    st.session_state.detection_count = get_current_detection_count()
    
correct_detection_numbers()

st.title("Baxter Detection")

source_selection = st.radio("Choose source:", ("Image Upload", "Live Camera"))

if source_selection == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])
    frame_placeholder = st.empty()

    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, channels="BGR")

        if st.button("Predict", key="predict_image"):
            detected_name = predict(image)
            if detected_name:
                st.success("Baxter detected in the image!")
                # Increment detection count
                st.session_state.detection_count += 1
                now = datetime.datetime.now()
                date_of_detection = now.strftime("%Y-%m-%d")
                time_of_detection = now.strftime("%H:%M:%S")
                save_detection(st.session_state.detection_count, detected_name, date_of_detection, time_of_detection)
            else:
                st.warning("No Baxter detected in the image.")

# Inside the Live Camera option
elif source_selection == "Live Camera":
    frame_width = 640
    frame_height = 480

    # Initialize camera capture if not already done
    if "cap" not in st.session_state:
        st.session_state.cap = initialize_window_and_capture(frame_width, frame_height)

    # Initialize session state variables if not already done
    if "run" not in st.session_state:
        st.session_state.run = False

    # Button to connect the model
    start_button = st.button("Connect Model", key="start_button")
    
    if start_button:
        st.success(f"Model Connected")
        st.session_state.run = True

    # Check if the model is connected or running
    if st.session_state.run:
        cap = st.session_state.cap

        # Capture the camera input
        frame = st.camera_input(label_visibility="hidden", label="hello", key="inputcamera")

        if frame:
            # Decode the image
            image = cv2.imdecode(np.frombuffer(frame.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform prediction
            detected_name = predict(rgb_frame)
            if detected_name:
                st.success(f"Detected object: {detected_name}")

                # Increment detection count
                st.session_state.detection_count += 1
                now = datetime.datetime.now()
                date_of_detection = now.strftime("%Y-%m-%d")
                time_of_detection = now.strftime("%H:%M:%S")
                save_detection(st.session_state.detection_count, detected_name, date_of_detection, time_of_detection)
            else:
                st.warning("No Baxter detected.")

        # Button to stop the camera
        if st.button("Stop Camera", key="stop_camera"):
            st.session_state.run = False
            cap.release()
            cv2.destroyAllWindows()
            st.stop()  # This stops the execution and removes the camera feed from the UI
