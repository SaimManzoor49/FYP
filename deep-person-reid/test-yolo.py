import cv2
from ultralytics import YOLO
import time
from datetime import datetime

def init_yolo():
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')  # Load the smallest YOLOv8 model for better speed
    return model

def init_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def log_detection(file_path, detections):
    """Log detection details to text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(file_path, 'a') as f:
        for detection in detections:
            # Extract features
            class_name = detection.names[int(detection.boxes.cls[0])]
            confidence = float(detection.boxes.conf[0])
            bbox = detection.boxes.xyxy[0].tolist()  # Convert bbox tensor to list
            
            # Format the log entry
            log_entry = (
                f"Timestamp: {timestamp}\n"
                f"Object: {class_name}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Bounding Box (x1,y1,x2,y2): {[round(x, 2) for x in bbox]}\n"
                f"------------------------\n"
            )
            f.write(log_entry)

def main():
    # Initialize model and webcam
    model = init_yolo()
    cap = init_webcam()
    
    # Create or clear the log file
    log_file = "detection_log.txt"
    with open(log_file, 'w') as f:
        f.write("YOLOv8 Object Detection Log\n========================\n")
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Run YOLOv8 inference
            results = model(frame, conf=0.5)  # Confidence threshold of 0.5
            
            # Process detections
            if len(results) > 0:
                # Visualize detections on frame
                annotated_frame = results[0].plot()
                
                # Log detections if any objects are detected
                if len(results[0].boxes) > 0:
                    log_detection(log_file, results)
                
                # Display the frame
                cv2.imshow('YOLOv8 Object Detection', annotated_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()