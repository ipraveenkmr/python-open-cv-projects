# pip install tensorflow opencv-python numpy

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import urllib.request
import os
import threading
import time

class AdvancedObjectDetector:
    def __init__(self):
        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')
        
        # COCO dataset class names (80 classes) + custom objects
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Additional objects we can detect with custom methods
        self.custom_objects = ['pen', 'pencil', 'marker', 'smartphone', 'tablet', 'headphones', 
                              'earbuds', 'charger', 'cable', 'glasses', 'sunglasses']
        
        # Enhanced object detection confidence threshold
        self.confidence_threshold = 0.3  # Lower threshold for better detection of small objects
        
        # Colors for different classes
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Load model
        self.model = None
        self.model_loaded = False
        self.load_model()
        
        # Face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def load_model(self):
        """Load TensorFlow Lite model for object detection"""
        try:
            print("Loading object detection model...")
            
            # Download model if not exists
            model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
            model_path = "detect.tflite"
            
            if not os.path.exists(model_path):
                print("Downloading model... This may take a moment.")
                import zipfile
                urllib.request.urlretrieve(model_url, "model.zip")
                with zipfile.ZipFile("model.zip", 'r') as zip_ref:
                    zip_ref.extractall(".")
                os.remove("model.zip")
            
            # Load TensorFlow Lite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.model_loaded = True
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback detection methods...")
            self.model_loaded = False
    
    def detect_faces(self, frame):
        """Detect faces using OpenCV Haar Cascades"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [x, y, x+w, y+h],
                'class': 'face',
                'confidence': 0.9
            })
        
        return detections
    
    def detect_objects_tflite(self, frame):
        """Detect objects using TensorFlow Lite model"""
        if not self.model_loaded:
            return []
        
        try:
            # Prepare input
            input_shape = self.input_details[0]['shape']
            height, width = input_shape[1], input_shape[2]
            
            # Resize and preprocess frame
            resized_frame = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(resized_frame, axis=0)
            
            if self.input_details[0]['dtype'] == np.uint8:
                input_data = input_data.astype(np.uint8)
            else:
                input_data = input_data.astype(np.float32)
                input_data = (input_data - 127.5) / 127.5
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get outputs
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
            
            # Process detections
            detections = []
            frame_height, frame_width = frame.shape[:2]
            
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:  # Lower threshold for better small object detection
                    # Convert normalized coordinates to pixel coordinates
                    y1 = int(boxes[i][0] * frame_height)
                    x1 = int(boxes[i][1] * frame_width)
                    y2 = int(boxes[i][2] * frame_height)
                    x2 = int(boxes[i][3] * frame_width)
                    
                    class_idx = int(classes[i])
                    if class_idx < len(self.class_names):
                        class_name = self.class_names[class_idx]
                        
                        # Special handling for phone detection
                        if class_name == 'cell phone':
                            class_name = 'smartphone'
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'class': class_name,
                            'confidence': scores[i]
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def detect_custom_objects(self, frame):
        """Detect custom objects like pens, glasses, etc. using shape and edge detection"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for thin objects like pens
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines (for pens, pencils)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            # Group nearby lines to detect pen-like objects
            pen_candidates = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Filter for pen-like objects (long, thin lines)
                if length > 80 and length < 300:
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # Create bounding box around the line
                    margin = 15
                    bbox = [min(x1,x2)-margin, min(y1,y2)-margin, 
                           max(x1,x2)+margin, max(y1,y2)+margin]
                    
                    pen_candidates.append({
                        'bbox': bbox,
                        'class': 'pen',
                        'confidence': 0.6,
                        'length': length,
                        'angle': angle
                    })
            
            # Remove overlapping detections
            if pen_candidates:
                # Simple non-maximum suppression for pen detections
                pen_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                final_pens = []
                
                for candidate in pen_candidates:
                    is_duplicate = False
                    for existing in final_pens:
                        # Check if bounding boxes overlap significantly
                        if self.calculate_overlap(candidate['bbox'], existing['bbox']) > 0.3:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate and len(final_pens) < 5:  # Limit to 5 pen detections
                        final_pens.append(candidate)
                
                detections.extend(final_pens)
        
        # Detect rectangular objects (smartphones, tablets when not detected by main model)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 2000 < area < 50000:  # Size range for phones/tablets
                # Approximate the contour to check if it's rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) == 4:  # Rectangular shape
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Check aspect ratio for phone-like objects
                    if 0.4 < aspect_ratio < 2.5:  # Phone/tablet aspect ratios
                        object_type = 'tablet' if area > 15000 else 'smartphone'
                        detections.append({
                            'bbox': [x, y, x+w, y+h],
                            'class': object_type,
                            'confidence': 0.5
                        })
        
        return detections
    
    def calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Get color for this class
            if class_name in self.class_names:
                color_idx = self.class_names.index(class_name)
                color = self.colors[color_idx]
            elif class_name == 'face':
                color = (0, 255, 0)  # Green for faces
            else:
                color = (255, 255, 255)  # White for unknown
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        print("Starting advanced object detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect objects using available methods
            all_detections = []
            
            # Method 1: TensorFlow Lite detection
            if self.model_loaded:
                tflite_detections = self.detect_objects_tflite(frame)
                all_detections.extend(tflite_detections)
            
            # Method 2: Face detection
            face_detections = self.detect_faces(frame)
            all_detections.extend(face_detections)
            
            # Method 3: Custom object detection (pens, etc.)
            custom_detections = self.detect_custom_objects(frame)
            all_detections.extend(custom_detections)
            
            # Draw all detections
            annotated_frame = self.draw_detections(frame, all_detections)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
            else:
                fps = 0
            
            if fps > 0:
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display detection count
            cv2.putText(annotated_frame, f"Objects: {len(all_detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Advanced Object Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")

# Additional utility class for better performance
class MultiThreadDetector(AdvancedObjectDetector):
    def __init__(self):
        super().__init__()
        self.detection_thread = None
        self.current_frame = None
        self.current_detections = []
        self.running = True
        
    def detection_worker(self):
        """Worker thread for object detection"""
        while self.running:
            if self.current_frame is not None:
                # Perform detection
                all_detections = []
                
                if self.model_loaded:
                    tflite_detections = self.detect_objects_tflite(self.current_frame)
                    all_detections.extend(tflite_detections)
                
                face_detections = self.detect_faces(self.current_frame)
                all_detections.extend(face_detections)
                
                custom_detections = self.detect_custom_objects(self.current_frame)
                all_detections.extend(custom_detections)
                
                self.current_detections = all_detections
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
    
    def run_threaded(self):
        """Run detection with threading for better performance"""
        print("Starting threaded object detection...")
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_worker)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Draw detections from background thread
            annotated_frame = self.draw_detections(frame, self.current_detections.copy())
            
            cv2.putText(annotated_frame, f"Objects: {len(self.current_detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Threaded Object Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose detection mode:")
    print("1. Standard Detection")
    print("2. Threaded Detection (Better Performance)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        detector = MultiThreadDetector()
        detector.run_threaded()
    else:
        detector = AdvancedObjectDetector()
        detector.run()