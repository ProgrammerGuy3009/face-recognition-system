from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from io import BytesIO
from pathlib import Path
import logging
from datetime import datetime

from config import API_CONFIG, DETECTION_CONFIG, EMBEDDING_CONFIG, MATCHING_CONFIG
from models.detector import FaceDetector
from models.embedder import FaceEmbedder
from models.matcher import FaceMatcher
from utils.preprocessing import FacePreprocessor
from utils.alignment import FaceAligner
from database.db_manager import DatabaseManager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Service",
    description="End-to-end face detection and recognition microservice",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
face_detector = FaceDetector(DETECTION_CONFIG)
face_embedder = FaceEmbedder(EMBEDDING_CONFIG)
face_matcher = FaceMatcher(MATCHING_CONFIG)
face_aligner = FaceAligner()
preprocessor = FacePreprocessor()
db_manager = DatabaseManager()

# Load gallery embeddings
try:
    gallery_data = db_manager.load_gallery()
    if gallery_data['embeddings'].shape[0] > 0:
        face_matcher.build_index(gallery_data['embeddings'], gallery_data['identities'])
        logger.info(f"✓ Loaded {len(gallery_data['identities'])} identities from database")
except Exception as e:
    logger.warning(f"Could not load gallery: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {
            "detector": "RetinaFace",
            "embedder": "ArcFace",
            "device": "cpu"
        }
    }


@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Detect faces
        detections = face_detector.detect(image)
        
        # Format response
        response_detections = []
        for i, det in enumerate(detections):
            response_detections.append({
                "face_id": i,
                "bbox": det['bbox'],
                "confidence": float(det['confidence'])
            })
        
        return {
            "detections": response_detections,
            "num_faces": len(detections),
            "image_shape": list(image.shape[:2])
        }
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...), threshold: float = Form(0.6)):
    try:
        # Read image
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Detect faces
        detections = face_detector.detect(image)
        
        recognitions = []
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            
            # Crop and align face
            face_crop = image[y1:y2, x1:x2]
            try:
                aligned_face = face_aligner.align(face_crop, detection)
                
                # Extract embedding
                embedding = face_embedder.extract_embedding(aligned_face)
                
                # Search gallery
                matches = face_matcher.search(embedding, threshold)
                
                recognitions.append({
                    "face_id": i,
                    "bbox": detection['bbox'],
                    "detected_confidence": float(detection['confidence']),
                    "matches": matches
                })
            except Exception as align_error:
                logger.warning(f"Could not align face {i}: {align_error}")
                recognitions.append({
                    "face_id": i,
                    "bbox": detection['bbox'],
                    "detected_confidence": float(detection['confidence']),
                    "matches": [],
                    "error": "Alignment failed"
                })
        
        return {"recognitions": recognitions}
    
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/add_identity")
async def add_identity(
    identity_name: str = Form(...),
    files: list = File(...)
):
    try:
        embeddings_list = []
        
        for file in files:
            contents = await file.read()
            image_array = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Detect and align face
            detections = face_detector.detect(image)
            if len(detections) == 0:
                continue
            
            detection = detections[0]  # Take first face
            x1, y1, x2, y2 = detection['bbox']
            face_crop = image[y1:y2, x1:x2]
            
            try:
                aligned_face = face_aligner.align(face_crop, detection)
                embedding = face_embedder.extract_embedding(aligned_face)
                embeddings_list.append(embedding)
            except Exception as e:
                logger.warning(f"Could not process image: {e}")
                continue
        
        if len(embeddings_list) == 0:
            raise ValueError("No valid faces detected in images")
        
        # Store in database
        avg_embedding = np.mean(embeddings_list, axis=0)
        db_manager.add_identity(identity_name, avg_embedding, len(embeddings_list))
        
        # Rebuild matcher index
        gallery_data = db_manager.load_gallery()
        face_matcher.build_index(gallery_data['embeddings'], gallery_data['identities'])
        
        return {
            "identity": identity_name,
            "enrolled_faces": len(embeddings_list),
            "message": "Identity added successfully"
        }
    
    except Exception as e:
        logger.error(f"Add identity error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/list_identities")
async def list_identities():
    try:
        identities = db_manager.list_identities()
        return {
            "identities": identities,
            "total_count": len(identities)
        }
    except Exception as e:
        logger.error(f"List identities error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/delete_identity/{identity_name}")
async def delete_identity(identity_name: str):
    try:
        db_manager.delete_identity(identity_name)
        
        # Rebuild matcher index
        gallery_data = db_manager.load_gallery()
        if gallery_data['embeddings'].shape[0] > 0:
            face_matcher.build_index(gallery_data['embeddings'], gallery_data['identities'])
        
        return {
            "message": f"Identity {identity_name} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Delete identity error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        # Read first image
        contents1 = await file1.read()
        image_array1 = np.frombuffer(contents1, np.uint8)
        image1 = cv2.imdecode(image_array1, cv2.IMREAD_COLOR)
        
        # Read second image
        contents2 = await file2.read()
        image_array2 = np.frombuffer(contents2, np.uint8)
        image2 = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)
        
        if image1 is None or image2 is None:
            raise ValueError("Could not decode images")
        
        # Detect and align faces
        detections1 = face_detector.detect(image1)
        detections2 = face_detector.detect(image2)
        
        if len(detections1) == 0 or len(detections2) == 0:
            raise ValueError("Could not detect face in one or both images")
        
        # Process first face
        x1, y1, x2, y2 = detections1[0]['bbox']
        face1_crop = image1[y1:y2, x1:x2]
        aligned_face1 = face_aligner.align(face1_crop, detections1[0])
        embedding1 = face_embedder.extract_embedding(aligned_face1)
        
        # Process second face
        x1, y1, x2, y2 = detections2[0]['bbox']
        face2_crop = image2[y1:y2, x1:x2]
        aligned_face2 = face_aligner.align(face2_crop, detections2[0])
        embedding2 = face_embedder.extract_embedding(aligned_face2)
        
        # Verify
        result = face_matcher.verify(embedding1, embedding2)
        
        return result
    
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        workers=API_CONFIG['workers'],
        log_level=API_CONFIG['log_level']
    )