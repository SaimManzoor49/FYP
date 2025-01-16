import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from collections import defaultdict
import cv2

class PersonReID:
    def __init__(self, yolo_model='yolov8n.pt', reid_model='osnet_x1_0', device='cuda'):
        self.device = device
        self.yolo = YOLO(yolo_model)
        self.extractor = FeatureExtractor(
            model_name=reid_model,
            device=device
        )


# ================================================================ CONSTANTS =============================================================
        
        
        # Initialize tracking components
        self.next_id = 1
        self.detection_history = {}
        self.waiting_detections = defaultdict(list)
        self.feature_db = {}
        self.db_path = Path('reid_database.json')
        self.load_database()
        
        # Detection count tracking
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False
        
        # # Constants 111.mp4
        # self.WAITING_FRAMES = 5 # 20
        # self.MAX_FEATURES_PER_ID = 5 # 10
        # self.MIN_BBOX_SIZE = 100
        # self.IOU_THRESHOLD = 0.9 # 0.3
        # self.FEATURE_MATCH_THRESHOLD = 0.7
        # self.MAX_FRAMES_MISSING = 40

        # Constants live iphone cam
        self.WAITING_FRAMES = 2 # 20
        self.MAX_FEATURES_PER_ID = 10 # 10
        self.MIN_BBOX_SIZE = 100
        self.IOU_THRESHOLD = 0.9 # 0.3
        self.FEATURE_MATCH_THRESHOLD = 0.8
        self.MAX_FRAMES_MISSING = 40

        # # test.mp4
        # self.WAITING_FRAMES = 35 # 20
        # self.MAX_FEATURES_PER_ID = 10 # 10
        # self.MIN_BBOX_SIZE = 100
        # self.IOU_THRESHOLD = 0.9 # 0.3
        # self.FEATURE_MATCH_THRESHOLD = 0.7
        # self.MAX_FRAMES_MISSING = 40

    def load_database(self):
        """Load the feature database from disk."""
        self.feature_db = {}  # Clear existing database
        self.next_id = 1      # Reset ID counter
        
        # Delete existing database file if it exists
        if self.db_path.exists():
            self.db_path.unlink()  # Delete the file
            
        # Create fresh empty database file
        try:
            with open(self.db_path, 'w') as f:
                json.dump({}, f)
        except Exception as e:
            self.feature_db = {}  # Ensure database is empty in case of error
            
        # Reset tracking variables
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False
        self.detection_history = {}
        self.waiting_detections = defaultdict(list)  # Ensure database is empty in case of error

    def save_database(self):
        """Save the feature database to disk."""
        try:
            serializable_db = {str(k): [v.tolist() for v in vectors] 
                             for k, vectors in self.feature_db.items()}
            with open(self.db_path, 'w') as f:
                json.dump(serializable_db, f)
        except Exception as e:
            print(f"Warning: Could not save database ({str(e)})")

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection + 1e-6)

    def extract_features(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            height, width = y2 - y1, x2 - x1
            
            if height < self.MIN_BBOX_SIZE or width < self.MIN_BBOX_SIZE:
                return None
                
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
                
            features = self.extractor(crop)
            features_np = features.cpu().numpy().flatten()
            features_normalized = features_np / np.linalg.norm(features_np)
            return features_normalized
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            return None

    def match_features(self, query_features, min_similarity=None):
        if min_similarity is None:
            min_similarity = self.FEATURE_MATCH_THRESHOLD
            
        max_similarity = 0
        matched_id = None
        
        for obj_id, stored_features in self.feature_db.items():
            for features in stored_features:
                similarity = np.dot(query_features, features)
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_id = obj_id
                    
        if max_similarity < min_similarity:
            return None, max_similarity
            
        return matched_id, max_similarity

    def update_feature_array(self, obj_id, new_features):
        if obj_id not in self.feature_db:
            self.feature_db[obj_id] = []
            
        should_add = True
        for existing_features in self.feature_db[obj_id]:
            similarity = np.dot(new_features, existing_features)
            if similarity > 0.95:
                should_add = False
                break
                
        if should_add:
            if len(self.feature_db[obj_id]) >= self.MAX_FEATURES_PER_ID:
                self.feature_db[obj_id].pop(0)
            self.feature_db[obj_id].append(new_features)

    def check_detection_count_change(self, current_count, frame_id):
        if current_count != self.previous_detection_count:
            self.is_in_waiting_period = True
            self.detection_change_frame = frame_id
            self.waiting_detections.clear()
            self.previous_detection_count = current_count
            return True
        return False

    def process_frame(self, frame, frame_id):
        results = []
        detections = self.yolo(frame, classes=0)[0]
        
        valid_detections = []
        for det in detections.boxes.data:
            x1, y1, x2, y2, conf, _ = det
            if conf < 0.3: #################### detection conf
                continue
            
            bbox = [x1, y1, x2, y2]
            features = self.extract_features(frame, bbox)
            if features is not None:
                valid_detections.append((bbox, features))

        current_detection_count = len(valid_detections)
        detection_count_changed = self.check_detection_count_change(current_detection_count, frame_id)
        
        for track_id in list(self.detection_history.keys()):
            self.detection_history[track_id]['frames_missing'] += 1
            if self.detection_history[track_id]['frames_missing'] > self.MAX_FRAMES_MISSING:
                del self.detection_history[track_id]

        if not self.feature_db and valid_detections:
            for bbox, features in valid_detections:
                new_id = self.next_id
                self.next_id += 1
                self.update_feature_array(new_id, features)
                self.detection_history[new_id] = {
                    'bbox': bbox,
                    'features': features,
                    'frames_missing': 0
                }
                results.append((bbox, new_id))
            return results
        
        if self.is_in_waiting_period:
            for bbox, features in valid_detections:
                detection_key = tuple(map(int, bbox))
                self.waiting_detections[detection_key].append({
                    'frame_id': frame_id,
                    'features': features,
                    'bbox': bbox
                })
                results.append((bbox, None))
                
            if frame_id - self.detection_change_frame >= self.WAITING_FRAMES:
                self.is_in_waiting_period = False
                
                for det_key, det_history in self.waiting_detections.items():
                    if len(det_history) >= self.WAITING_FRAMES * 0.8:
                        avg_features = np.mean([d['features'] for d in det_history], axis=0)
                        avg_features = avg_features / np.linalg.norm(avg_features)
                        
                        matched_id, similarity = self.match_features(avg_features)
                        if matched_id is not None:
                            self.update_feature_array(matched_id, avg_features)
                        else:
                            matched_id = self.next_id
                            self.next_id += 1
                            self.update_feature_array(matched_id, avg_features)
                            
                        latest_detection = det_history[-1]
                        self.detection_history[matched_id] = {
                            'bbox': latest_detection['bbox'],
                            'features': latest_detection['features'],
                            'frames_missing': 0
                        }
                        results.append((latest_detection['bbox'], matched_id))
                
                self.waiting_detections.clear()
        else:
            for bbox, features in valid_detections:
                best_iou = 0
                matched_id = None
                
                for track_id, track_info in self.detection_history.items():
                    iou = self.calculate_iou(bbox, track_info['bbox'])
                    if iou > self.IOU_THRESHOLD and iou > best_iou:
                        best_iou = iou
                        matched_id = track_id
                
                if matched_id is not None:
                    self.detection_history[matched_id] = {
                        'bbox': bbox,
                        'features': features,
                        'frames_missing': 0
                    }
                    self.update_feature_array(matched_id, features)
                    results.append((bbox, matched_id))
                else:
                    matched_id, similarity = self.match_features(features)
                    if matched_id is not None:
                        self.detection_history[matched_id] = {
                            'bbox': bbox,
                            'features': features,
                            'frames_missing': 0
                        }
                        self.update_feature_array(matched_id, features)
                        results.append((bbox, matched_id))
                    else:
                        new_id = self.next_id
                        self.next_id += 1
                        self.update_feature_array(new_id, features)
                        self.detection_history[new_id] = {
                            'bbox': bbox,
                            'features': features,
                            'frames_missing': 0
                        }
                        results.append((bbox, new_id))
        
        return results

def main():
    reid_system = PersonReID(
        yolo_model='./models/yolov8n.onnx',
        reid_model='osnet_ain_x1_0',
        device='cpu'
    )

    cap = cv2.VideoCapture(1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = reid_system.process_frame(frame, frame_id)
        frame_id += 1
        
        for bbox, obj_id in results:
            x1, y1, x2, y2 = map(int, bbox)
            if obj_id is not None:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {obj_id}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "Waiting...", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Detections: {len([r for r in results if r[1] is not None])}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # out.write(frame)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    reid_system.save_database()

if __name__ == "__main__":
    main()