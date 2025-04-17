import streamlit as st
import cv2
import os
import tempfile

# Function to detect and save number plates from the video
def detect_number_plate(video_file):
    output_dir = 'detected_plates'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load pre-trained Haar Cascade for number plate detection
    plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

    # Capturing video
    video_capture = cv2.VideoCapture(video_file)

    plate_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect number plates
        plates = plate_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around plate
            plate_region = frame[y:y+h, x:x+w]  # Crop the plate region

            # Save the detected plate
            plate_count += 1
            cv2.imwrite(os.path.join(output_dir, f'plate_{plate_count}.jpg'), plate_region)

    # Release the capture
    video_capture.release()

# Streamlit application
st.title("Number Plate Detection System")
st.write("Upload a video file to detect and save number plates.")

# File uploader
uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.video(temp_file_path)  # Display the video

    if st.button("Detect Number Plates"):
        with st.spinner("Detecting number plates..."):
            detect_number_plate(temp_file_path)  # Call the detection function
            
        st.success("Detection complete! Check the 'detected_plates' folder for saved images.")