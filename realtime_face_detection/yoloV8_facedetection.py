import cv2
from ultralytics import YOLO

def live_yolov8_face_detection(model_name: str = 'yolov8n-face.pt', conf_threshold: float = 0.5, iou_threshold: float = 0.7, webcam_index: int = 0):
    """
    Performs real-time face detection using a YOLOv8-face model on a webcam stream.

    Args:
        model_name (str): The YOLOv8 face detection model name (e.g., 'yolov8n-face.pt', 'yolov8l-face.pt').
                          The weights file will download automatically on first run.
        conf_threshold (float): Confidence threshold for detection (0.0 to 1.0).
        iou_threshold (float): IoU threshold for Non-Maximum Suppression (0.0 to 1.0).
        webcam_index (int): Index of the webcam to use (0 is typically the default camera).
    """
    try:
        # Load the specified YOLOv8 model
        print(f"Loading model: {model_name}...")
        # model = YOLO(model_name)
        model = YOLO()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Open the default webcam
    cap = cv2.VideoCapture(webcam_index)

    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {webcam_index}.")
        return

    print("--- Press 'q' to exit the detection window ---")

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break

        # Run face detection inference on the frame
        # 'stream=True' is recommended for video to process frames efficiently
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False  # Suppress prediction output for every frame
        )
        
        # Process results: results[0].plot() draws bounding boxes and labels on the image
        annotated_frame = results[0].plot()

        # Display the resulting frame
        cv2.imshow("YOLOv8 Face Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed closed.")

if __name__ == '__main__':
    live_yolov8_face_detection()