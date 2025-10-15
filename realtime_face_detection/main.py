# from haar_facedetection import capture_and_process_webcam_feed
# from dnn_facedetection import capture_and_process_webcam_feed
from yoloV8_facedetection import live_yolov8_face_detection

# Example usage (run this in your main file)
if __name__ == '__main__':
    # capture_and_process_webcam_feed(target_width=224, target_height=224)
    live_yolov8_face_detection()