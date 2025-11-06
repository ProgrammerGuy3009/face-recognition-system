import cv2
import numpy as np
import onnxruntime as rt
from pathlib import Path
from typing import List, Tuple
import torch
from retinaface import RetinaFace
from config import DETECTION_CONFIG

class FaceDetector:
    def __init__(self, config=None):
        self.config = config or DETECTION_CONFIG
        self.device = self.config.get("device", "cpu")
        self.model = None
        self.onnx_session = None
        self._load_model()
    
    def _load_model(self):
        backbone = self.config.get("backbone", "mobilenetv1")
        
        if self.config.get("use_onnx", False):
            self._load_onnx_model(backbone)
        else:
            self._load_pytorch_model(backbone)
    
    def _load_pytorch_model(self, backbone: str):
        try:
            self.model = RetinaFace(backbone=backbone, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ RetinaFace-{backbone} model loaded on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load RetinaFace model: {e}")
    
    def _load_onnx_model(self, backbone: str):
        onnx_path = Path(__file__).parent.parent / "models_weights" / f"retinaface_{backbone}.onnx"
        
        if not onnx_path.exists():
            print(f"⚠ ONNX model not found at {onnx_path}")
            print("Downloading ONNX weights...")
            # Download ONNX weights if not present
            self._download_onnx_weights(backbone, onnx_path)
        
        try:
            self.onnx_session = rt.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            print(f"✓ ONNX RetinaFace-{backbone} model loaded on CPU")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def detect(self, image: np.ndarray) -> List[Tuple]:
        if image is None or image.size == 0:
            return []
        
        # Resize for processing if needed
        original_shape = image.shape
        if self.config.get("use_onnx", False):
            detections = self._detect_onnx(image)
        else:
            detections = self._detect_pytorch(image)
        
        # Apply post-processing
        detections = self._apply_nms(detections)
        detections = self._filter_small_faces(detections)
        
        return detections
    
    def _detect_pytorch(self, image: np.ndarray) -> List[Tuple]:
        with torch.no_grad():
            try:
                # RetinaFace returns dict with face information
                faces = self.model.detect_faces(image)
                detections = []
                
                for face in faces:
                    det = {
                        'bbox': face['box'],
                        'confidence': face['confidence'],
                        'landmarks': face.get('landmarks', []),
                    }
                    detections.append(det)
                
                return detections
            except Exception as e:
                print(f"Error during detection: {e}")
                return []
    
    def _detect_onnx(self, image: np.ndarray) -> List[Tuple]:
        try:
            # Prepare input
            h, w = image.shape[:2]
            scale = 1.0
            
            # ONNX expects specific input format
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (640, 640),
                (104, 117, 123), swapRB=False, crop=False
            )
            
            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name
            output_names = [output.name for output in self.onnx_session.get_outputs()]
            
            results = self.onnx_session.run(output_names, {input_name: blob})
            
            # Parse results (depends on ONNX model output format)
            detections = self._parse_onnx_output(results, image.shape)
            return detections
        except Exception as e:
            print(f"Error during ONNX detection: {e}")
            return []
    
    def _parse_onnx_output(self, outputs, image_shape) -> List[Tuple]:
        detections = []
        h, w = image_shape[:2]
        
        # Typical ONNX output: [boxes, scores, landmarks, etc.]
        # This varies by model - adjust based on your exported ONNX format
        try:
            for i, output in enumerate(outputs):
                if output.shape[-1] >= 4:  # Has bounding box
                    boxes = output
                    for box in boxes:
                        x1, y1, x2, y2 = box[:4]
                        conf = box[4] if len(box) > 4 else 0.5
                        
                        if conf > self.config['confidence_threshold']:
                            det = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'landmarks': [],
                            }
                            detections.append(det)
        except Exception as e:
            print(f"Error parsing ONNX output: {e}")
        
        return detections
    
    def _apply_nms(self, detections: List[Tuple]) -> List[Tuple]:
        if not detections:
            return []
        
        # Extract boxes and confidences
        boxes = np.array([d['bbox'] for d in detections])
        confidences = np.array([d['confidence'] for d in detections])
        
        # OpenCV NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(confidences)[::-1]
        
        keep = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > self.config['nms_threshold'])[0])))
        
        return [detections[i] for i in keep]
    
    def _filter_small_faces(self, detections: List[Tuple]) -> List[Tuple]:
        min_size = self.config.get('min_face_size', 16)
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            
            if w >= min_size and h >= min_size:
                filtered.append(det)
        
        return filtered
    
    def visualize(self, image: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            text = f"Face: {conf:.2f}"
            cv2.putText(output, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw landmarks if available
            if det.get('landmarks'):
                for point in det['landmarks']:
                    cv2.circle(output, tuple(map(int, point)), 3, (0, 0, 255), -1)
        
        return output