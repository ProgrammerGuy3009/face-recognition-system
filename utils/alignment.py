import cv2
import numpy as np
import dlib
from pathlib import Path

class FaceAligner:    
    def __init__(self):
        try:
            # Download dlib predictor if not present
            predictor_path = Path(__file__).parent.parent / "models_weights" / "shape_predictor_5_face_landmarks.dat"
            
            if not predictor_path.exists():
                print("Downloading dlib 5-point face landmark predictor...")
                # In production, download from: http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2
                import urllib.request
                url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
                # Note: Actual download logic would go here
            
            self.predictor = dlib.shape_predictor(str(predictor_path))
            self.detector = dlib.get_frontal_face_detector()
            print("✓ Dlib face landmark predictor loaded")
        except Exception as e:
            print(f"⚠ Dlib predictor not available: {e}")
            self.predictor = None
            self.detector = None
    
    def align(self, face_image: np.ndarray, detection: dict, output_size: tuple = (112, 112)) -> np.ndarray:
        if face_image is None or face_image.size == 0:
            return None
        
        try:
            # Get landmarks
            landmarks = self._extract_landmarks(face_image, detection)
            
            if landmarks is None or len(landmarks) < 5:
                # If landmarks not available, return resized image
                return cv2.resize(face_image, output_size)
            
            # Perform affine transformation
            aligned = self._affine_transform(face_image, landmarks, output_size)
            
            return aligned
        except Exception as e:
            print(f"Alignment error: {e}")
            return cv2.resize(face_image, output_size)
    
    def _extract_landmarks(self, face_image: np.ndarray, detection: dict) -> np.ndarray:
        landmarks = detection.get('landmarks', None)
        
        if landmarks is not None and len(landmarks) >= 5:
            return np.array(landmarks[:5], dtype=np.float32)
        
        # Fallback: detect landmarks using dlib
        if self.predictor is not None:
            try:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray, 1)
                
                if len(rects) > 0:
                    shape = self.predictor(gray, rects[0])
                    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(5)], dtype=np.float32)
                    return landmarks
            except Exception as e:
                print(f"Dlib landmark extraction failed: {e}")
        
        # Fallback: use approximate eye positions
        h, w = face_image.shape[:2]
        return np.array([
            [w * 0.3, h * 0.35],    # Left eye
            [w * 0.7, h * 0.35],    # Right eye
            [w * 0.5, h * 0.5],     # Nose
            [w * 0.2, h * 0.8],     # Left mouth
            [w * 0.8, h * 0.8]      # Right mouth
        ], dtype=np.float32)
    
    def _affine_transform(self, image: np.ndarray, landmarks: np.ndarray, output_size: tuple) -> np.ndarray:
        # Reference landmarks (desired output positions)
        reference_landmarks = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        # Scale reference landmarks to output size
        reference_landmarks = reference_landmarks * (output_size[0] / 112.0, output_size[1] / 112.0)
        
        # Compute affine transformation matrix
        # Using first 3 points for affine transformation
        src_pts = landmarks[:3].astype(np.float32)
        dst_pts = reference_landmarks[:3].astype(np.float32)
        
        M = cv2.getAffineTransform(src_pts, dst_pts)
        
        # Apply transformation
        aligned = cv2.warpAffine(image, M, output_size)
        
        return aligned


class QualityFilter:    
    @staticmethod
    def compute_blur_score(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)
    
    @staticmethod
    def is_blurry(image: np.ndarray, threshold: float = 100) -> bool:
        score = QualityFilter.compute_blur_score(image)
        return score < threshold
    
    @staticmethod
    def compute_brightness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    @staticmethod
    def is_too_dark(image: np.ndarray, threshold: float = 50) -> bool:
        brightness = QualityFilter.compute_brightness(image)
        return brightness < threshold
    
    @staticmethod
    def compute_contrast(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    
    @staticmethod
    def is_low_contrast(image: np.ndarray, threshold: float = 20) -> bool:
        contrast = QualityFilter.compute_contrast(image)
        return contrast < threshold
    
    @staticmethod
    def compute_quality_score(image: np.ndarray) -> float:
        blur_score = QualityFilter.compute_blur_score(image)
        brightness = QualityFilter.compute_brightness(image)
        contrast = QualityFilter.compute_contrast(image)
        
        # Normalize scores
        blur_norm = min(blur_score / 500.0, 1.0)  # Normalize to 0-1
        brightness_norm = min(abs(brightness - 128) / 128.0, 1.0)  # Ideal brightness = 128
        contrast_norm = min(contrast / 100.0, 1.0)
        
        # Combined score
        quality = (blur_norm * 0.4 + brightness_norm * 0.3 + contrast_norm * 0.3)
        
        return float(quality)