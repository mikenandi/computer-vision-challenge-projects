import cv2
import numpy as np

# Define the path to your Haar Cascade XML file
# Make sure this file is in the same directory or provide the full path
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'

def capture_and_process_webcam_feed(target_width=224, target_height=224):
    """
    Initializes webcam, performs Haar Cascade face detection,
    and processes the detected face frames for model input.
    """
    # 1. Load the Haar Cascade Classifier (Done only once)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    if face_cascade.empty():
        print(f"Error: Could not load Haar Cascade file at {HAAR_CASCADE_PATH}")
        return

    # 2. Initialize the webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam feed started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        # Convert the frame to grayscale for detection
        # Haar Cascades perform better on grayscale images
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Perform Face Detection
        # detectMultiScale returns a list of bounding boxes (x, y, w, h)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,         # How much the image size is reduced at each image scale
            minNeighbors=5,          # How many neighbors each candidate rectangle should have
            minSize=(30, 30),        # Minimum possible object size
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 4. Process Detected Faces (Resize and Normalize)
        processed_input = [] # List to hold the normalized face images

        for (x, y, w, h) in faces:
            # Draw a green rectangle around the detected face on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the detected face region from the original color frame
            face_crop = frame[y:y + h, x:x + w]

            # --- Resizing and Normalization ---
            # 2. Resize the face crop
            resized_face = cv2.resize(face_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

            # 3. Normalize the face frame (Scale 0-255 to 0.0-1.0)
            normalized_face = resized_face.astype('float32') / 255.0

            # Add the processed face to the list
            processed_input.append(normalized_face)

            # NOTE: You would typically pass 'normalized_face' to your model here

        # Display the frame with the detection boxes
        cv2.imshow('Face Detection Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")