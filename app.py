import streamlit as st
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
from roboflow import Roboflow
import time
from collections import defaultdict
import datetime
import os
import torch

def get_device():
  """
  This function checks for GPU availability and returns an appropriate device object.
  """
  if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Use first available GPU
    print("Using GPU for computations.")
  else:
    device = torch.device("cpu")
    print("Using CPU for computations (no GPU available).")
  return device

# Load the model on app startup (assuming it's lightweight)
model = YOLO('best.pt').to(get_device())  # Load model with device assignment

def predict(image):
    """Performs prediction on the given image and checks for 'Baxter' detection."""
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)

    if len(detections) == 0:
        return None  # No detections

    for detection in detections:
        box, score, class_id = detection[:3]  # Adjust based on your output structure
        class_id = int(class_id)  # Ensure class_id is an integer
        detected_name = model.names[class_id]  # Get the class name
        if detected_name == "Baxter":
            return "A Baxter is detected!"  # Baxter detected

    return None  # No Baxter detected





def initialize_window_and_capture(window_width, window_height):
  """
  Initializes the video capture object and background image.
  Attempts to open the default camera (source 0) first.
  Falls back to the external camera (source 1) if unavailable.
  """
  # Adjust screen dimensions to match your needs (optional)
  screen_width = 1280  
  screen_height = 720

  # Try opening default camera (source 0)
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Error opening default camera (source 0). Using external camera (source 1).")
    cap = cv2.VideoCapture(1)

  cap.set(3, window_width)
  cap.set(4, window_height)
  cap.set(cv2.CAP_PROP_EXPOSURE, -3)
  frame_rate = 30
  cap.set(cv2.CAP_PROP_FPS, frame_rate)

  # ... rest of the code for background image and window creation (optional)

  return cap


st.title("Baxter Detection")

# Radio buttons for user selection (image upload or camera)
source_selection = st.radio("Choose source:", ("Image Upload", "Live Camera"))

if source_selection == "Image Upload":
  uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])

  if uploaded_file is not None:
   image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
   st.image(image, channels="BGR")

  if st.button("Predict", key="predict_image"):  # Add key="predict_image"
    detected_name = predict(image)
    if detected_name:
      st.success("Baxter detected in the image!")
    else:
      st.warning("No Baxter detected in the image.")
      
elif source_selection == "Live Camera":
  # Set frame width and height (adjust as needed)
  frame_width = 640
  frame_height = 480

  # Initialize camera capture
  cap = initialize_window_and_capture(frame_width, frame_height)

  # Display live camera feed
  run = st.button("Run Camera")
  while run:
   success, frame = cap.read()
   if not success:
    print("Ignoring empty camera frame.")
    continue

   # Flip the frame horizontally
   frame = cv2.flip(frame, 1)

   # Perform prediction on the frame
  detected_name = predict(frame)
  if detected_name:
    st.write("Detected object:", detected_name)
  else:
    st.write("No objects detected.")

  # Convert frame to RGB for Streamlit display
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  st.image(rgb_frame, channels="RGB")

  # Stop camera feed on button press
  if st.button("Stop", key="stop_camera"):  # Add key="stop_camera"
    run = False

  # Release camera resources
  if cap is not None:
    cap.release()
  cv2.destroyAllWindows()