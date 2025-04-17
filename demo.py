from track import *
import tempfile
import cv2
import torch
import streamlit as st
import os
from speed_estimation import estimate_speed  # Importing the speed estimation function

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

if __name__ == '__main__':
    st.title('Vehicle Detection and Counting,Speed Estimation And Number Plate Detection')
    st.markdown('<h3 style="color: red"> with YOLOv5, Deep SORT Vehicle Counting, Speed Estimation  And Number Plate Detection </h3>', unsafe_allow_html=True)

    # Upload video
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi'])

    if video_file_buffer:
        st.sidebar.text('Input video')
        st.sidebar.video(video_file_buffer)
        temp_video_path = os.path.join('videos', video_file_buffer.name)
        with open(temp_video_path, 'wb') as f:
            f.write(video_file_buffer.getbuffer())

    # Sidebar settings
    st.sidebar.markdown('---')
    st.sidebar.title('Settings')
    custom_class = st.sidebar.checkbox('Custom classes')
    assigned_class_id = [0, 1, 2, 3]
    names = ['car', 'motorcycle', 'truck', 'bus']

    if custom_class:
        assigned_class_id = []
        assigned_class = st.sidebar.multiselect('Select custom classes', list(names))
        for each in assigned_class:
            assigned_class_id.append(names.index(each))

    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.5)
    line = st.sidebar.number_input('Line position', min_value=0.0, max_value=1.0, value=0.6, step=0.1)

    status = st.empty()
    stframe = st.empty()
    if video_file_buffer is None:
        status.markdown('<font size="4"> **Status:** Waiting for input </font>', unsafe_allow_html=True)
    else:
        status.markdown('<font size="4"> **Status:** Ready </font>', unsafe_allow_html=True)

    # Vehicle counters and speed display
    car, bus, truck, motor = st.columns(4)
    with car:
        st.markdown('**Car**')
        car_text = st.markdown('__')
    with bus:
        st.markdown('**Bus**')
        bus_text = st.markdown('__')
    with truck:
        st.markdown('**Truck**')
        truck_text = st.markdown('__')
    with motor:
        st.markdown('**Motorcycle**')
        motor_text = st.markdown('__')

    fps, _, _, speed = st.columns(4)
    with fps:
        st.markdown('**FPS**')
        fps_text = st.markdown('__')
    with speed:
        st.markdown('** Vehicle speed**')
        speed_text = st.markdown('__')

    track_button = st.sidebar.button('START')
    number_plate_button = st.sidebar.button('DETECT NUMBER PLATES')
    if video_file_buffer:
        if number_plate_button:
            with st.spinner("Detecting number plates..."):
                detect_number_plate(temp_video_path)  # Call the detection function
            st.success("Detection complete! Check the 'detected_plates' folder for saved images.")
            for img_file in os.listdir('detected_plates'):
                img_path = os.path.join('detected_plates', img_file)
                st.image(img_path, caption=img_file)

    if track_button and video_file_buffer:
        reset()
        opt = parse_opt()
        opt.conf_thres = confidence
        opt.source = temp_video_path

        status.markdown('<font size="4"> **Status:** Running... </font>', unsafe_allow_html=True)

        # Creating output directory for detected plates
        output_dir = "detected_plates"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with torch.no_grad():
            for frame in detect(opt, stframe, car_text, bus_text, truck_text, motor_text, line, fps_text, speed_text, assigned_class_id):
                # Write the frame to a temporary image to process number plate detection
                cv2.imwrite('temp_frame.jpg', frame)
                
                st.image(frame, caption='Processed Frame', use_container_width=True)  # Show the processed frame

        status.markdown('<font size="4"> **Status:** Finished! </font>', unsafe_allow_html=True)
        # Clean up
        os.remove('temp_frame.jpg')
        # Optionally, you can clear detected plates after displaying
        for img_file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, img_file))
        os.rmdir(output_dir)
        os.remove(temp_video_path)