import cv2
import os

# Ensure to have your output directory
output_dir = 'detected_plates'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load pre-trained Haar Cascade for number plate detection
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Capturing video
video_capture = cv2.VideoCapture('sample.mp4')

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

    # Display the video with detected plates
    #cv2.imshow('Number Plate Detection', frame)

    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()