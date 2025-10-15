import cv2
import numpy as np

# --- DNN Model Configuration ---
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"
CONFIG_FILE = "deploy.prototxt"
CONFIDENCE_THRESHOLD = 0.5 

def capture_and_process_webcam_feed(target_width=224, target_height=224):
    """
    Initializes webcam, performs DNN face detection, and processes 
    the detected face frames.
    """
    # 1. Load the DNN model (Done only once)
    try:
        net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)
    except cv2.error as e:
        print(f"Error loading DNN model: {e}")
        print("Ensure you have 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' in the correct path.")
        return

    # 2. Initialize the webcam feed
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("DNN Face Detection started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        (h, w) = frame.shape[:2]

        # 3. Create a blob from the frame
        # The DNN model expects a specific input format (300x300 BGR image, mean subtraction)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),  # Resize to 300x300
            1.0,                             # Scale factor
            (300, 300),                      # New spatial size
            (104.0, 177.0, 123.0)            # Mean subtraction for BGR channels (ImageNet means)
        )

        # 4. Pass the blob through the network and get detections
        net.setInput(blob)
        detections = net.forward()

        # 5. Loop over the detections and process faces
        processed_input = []
        
        # Detections is a 4D array: [1, 1, N, 7]
        # N is the number of detections, the last 7 columns are:
        # [0, 0, confidence, x1, y1, x2, y2] (coordinates are normalized 0-1)
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > CONFIDENCE_THRESHOLD:
                # Get the normalized coordinates and scale them back to the original image dimensions
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and confidence on the original frame
                text = "{:.2f}%".format(confidence * 100)
                y_text = startY - 10 if startY - 10 > 10 else startY + 10
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                # --- Extract, Resize, and Normalize Face for Model Input ---
                # Crop the detected face
                face_crop = frame[startY:endY, startX:endX]
                
                # Check for valid crop (in case bounding box is out of frame)
                if face_crop.any(): 
                    # Resize the face crop to the desired TARGET_WIDTH x TARGET_HEIGHT
                    resized_face = cv2.resize(face_crop, (target_width, target_height), interpolation=cv2.INTER_AREA)

                    # Normalize the face frame (Scale 0-255 to 0.0-1.0)
                    normalized_face = resized_face.astype('float32') / 255.0

                    processed_input.append(normalized_face)
                    # NOTE: This is where you would pass 'normalized_face' to your classification model

        # Display the frame with the detection boxes
        cv2.imshow('DNN Face Detection Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")
