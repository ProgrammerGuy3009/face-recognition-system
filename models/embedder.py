import cv2
import numpy as np
import onnxruntime as rt
from pathlib import Path
from typing import Tuple
import torch
from config import EMBEDDING_CONFIG

class FaceEmbedder:
    def __init__(self, config=None):
        self.config = config or EMBEDDING_CONFIG
        self.device = self.config.get("device", "cpu")
        self.model = None
        self.onnx_session = None
        self._load_model()
    
    def _load_model(self):
        model_name = self.config.get("model_name", "arcface")
        
        if self.config.get("use_onnx", False):
            self._load_onnx_model(model_name)
        else:
            self._load_pytorch_model(model_name)
    
    def _load_pytorch_model(self, model_name: str):
        try:
            from insightface.app import FaceAnalysis
            
            self.face_app = FaceAnalysis(
                name='buffalo_m',  # Pre-trained model
                providers=['CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0)
            print(f"✓ InsightFace model loaded (ArcFace embeddings)")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def _load_onnx_model(self, model_name: str):
        onnx_path = Path(__file__).parent.parent / "models_weights" / f"{model_name}.onnx"
        
        try:
            self.onnx_session = rt.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            print(f"✓ ONNX {model_name} model loaded on CPU")
        except Exception as e:
            print(f"⚠ ONNX model not found: {e}")
            self._load_pytorch_model(model_name)
    
    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        if face_image is None or face_image.size == 0:
            return np.zeros(self.config['embedding_dim'], dtype=np.float32)
        
        # Ensure correct input size
        if face_image.shape[:2] != (112, 112):
            face_image = cv2.resize(face_image, (112, 112))
        
        # Normalize image
        face_image = self._preprocess(face_image)
        
        if self.config.get("use_onnx", False):
            embedding = self._extract_onnx(face_image)
        else:
            embedding = self._extract_pytorch(face_image)
        
        # Normalize embedding to unit vector (important for cosine similarity)
        embedding = self._normalize_embedding(embedding)
        
        return embedding
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply normalization
        mean = np.array(self.config['normalize_mean'])
        std = np.array(self.config['normalize_std'])
        image = (image - mean) / std
        
        # Convert to (C, H, W) format for model
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        return image.astype(np.float32)
    
    def _extract_pytorch(self, face_image: np.ndarray) -> np.ndarray:
        try:
            with torch.no_grad():
                # Using InsightFace which internally uses PyTorch
                face_tensor = torch.from_numpy(face_image * 255).to(self.device)
                embeddings = self.face_app.get_feat(face_tensor)
                return embeddings
        except Exception as e:
            print(f"Error during PyTorch embedding extraction: {e}")
            return np.zeros(self.config['embedding_dim'], dtype=np.float32)
    
    def _extract_onnx(self, face_image: np.ndarray) -> np.ndarray:
        try:
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            
            embedding = self.onnx_session.run(
                [output_name],
                {input_name: face_image}
            )[0]
            
            return embedding.flatten().astype(np.float32)
        except Exception as e:
            print(f"Error during ONNX embedding extraction: {e}")
            return np.zeros(self.config['embedding_dim'], dtype=np.float32)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def batch_extract(self, face_images: list) -> np.ndarray:
        embeddings = []
        for face_img in face_images:
            emb = self.extract_embedding(face_img)
            embeddings.append(emb)
        
        return np.array(embeddings)


class EmbeddingCache:
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, image_hash: str) -> np.ndarray:
        return self.cache.get(image_hash, None)
    
    def put(self, image_hash: str, embedding: np.ndarray):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[image_hash] = embedding
    
    def clear(self):
        self.cache.clear()