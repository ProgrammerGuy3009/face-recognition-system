# API_DOCUMENTATION.md - FastAPI Swagger Documentation

## Face Recognition Service API Reference

**Base URL**: `http://localhost:8000`
**Version**: 1.0.0
**Protocol**: REST + JSON

---

## Endpoints

### 1. Health Check

**Endpoint**: `GET /health`

**Description**: Check service health and loaded models

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-05T14:30:00.000Z",
  "models": {
    "detector": "RetinaFace",
    "embedder": "ArcFace",
    "device": "cpu"
  }
}
```

**Status Codes**: 
- 200: OK
- 503: Service unavailable

---

### 2. Detect Faces

**Endpoint**: `POST /detect`

**Description**: Detect all faces in an image

**Request**:
```
Content-Type: multipart/form-data

Parameters:
- file (File, required): Image file (JPEG, PNG)
```

**Response**:
```json
{
  "detections": [
    {
      "face_id": 0,
      "bbox": [50, 80, 200, 280],
      "confidence": 0.95
    }
  ],
  "num_faces": 1,
  "image_shape": [480, 640]
}
```

**Face Object**:
- `face_id` (int): Face index in image
- `bbox` (list): [x1, y1, x2, y2] bounding box coordinates
- `confidence` (float): Detection confidence [0-1]

**Error Responses**:
```json
{
  "detail": "Could not decode image"
}
```

**cURL Example**:
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/detect
```

---

### 3. Recognize Faces

**Endpoint**: `POST /recognize`

**Description**: Detect faces and recognize identities from gallery

**Request**:
```
Content-Type: multipart/form-data

Parameters:
- file (File, required): Image file
- threshold (float, optional): Similarity threshold [0-1] (default: 0.6)
```

**Response**:
```json
{
  "recognitions": [
    {
      "face_id": 0,
      "bbox": [50, 80, 200, 280],
      "detected_confidence": 0.95,
      "matches": [
        {
          "identity": "john_doe",
          "similarity": 0.85,
          "rank": 1
        },
        {
          "identity": "jane_smith",
          "similarity": 0.72,
          "rank": 2
        }
      ]
    }
  ]
}
```

**Match Object**:
- `identity` (str): Registered identity name
- `similarity` (float): Cosine similarity score [0-1]
- `rank` (int): Rank in results

**cURL Example**:
```bash
curl -X POST -F "file=@test_image.jpg" \
     -F "threshold=0.6" \
     http://localhost:8000/recognize
```

---

### 4. Verify Faces

**Endpoint**: `POST /verify`

**Description**: Verify if two face images belong to the same person

**Request**:
```
Content-Type: multipart/form-data

Parameters:
- file1 (File, required): First face image
- file2 (File, required): Second face image
```

**Response**:
```json
{
  "match": true,
  "similarity": 0.87,
  "distance": 0.13
}
```

**Result Fields**:
- `match` (bool): True if faces match (similarity >= threshold)
- `similarity` (float): Cosine similarity [0-1]
- `distance` (float): Cosine distance [0-1]

**cURL Example**:
```bash
curl -X POST \
     -F "file1=@person_a_1.jpg" \
     -F "file2=@person_a_2.jpg" \
     http://localhost:8000/verify
```

---

### 5. Add Identity

**Endpoint**: `POST /add_identity`

**Description**: Enroll new person with face images

**Request**:
```
Content-Type: multipart/form-data

Parameters:
- identity_name (str, required): Name of person to enroll
- files (Files, required): 1+ face images for enrollment
```

**Response**:
```json
{
  "identity": "john_doe",
  "enrolled_faces": 3,
  "message": "Identity added successfully"
}
```

**Requirements**:
- Minimum 1 face image
- Faces must be clearly visible
- Recommended: 3-5 images from different angles

**cURL Example**:
```bash
curl -X POST \
     -F "identity_name=john_doe" \
     -F "files=@john_1.jpg" \
     -F "files=@john_2.jpg" \
     -F "files=@john_3.jpg" \
     http://localhost:8000/add_identity
```

---

### 6. List Identities

**Endpoint**: `GET /list_identities`

**Description**: Get all registered identities in gallery

**Response**:
```json
{
  "identities": [
    {
      "id": 1,
      "name": "john_doe",
      "num_samples": 3,
      "created_at": "2024-11-05T10:00:00",
      "updated_at": "2024-11-05T12:30:00"
    },
    {
      "id": 2,
      "name": "jane_smith",
      "num_samples": 2,
      "created_at": "2024-11-04T15:20:00",
      "updated_at": "2024-11-04T15:20:00"
    }
  ],
  "total_count": 2
}
```

**cURL Example**:
```bash
curl http://localhost:8000/list_identities
```

---

### 7. Delete Identity

**Endpoint**: `DELETE /delete_identity/{identity_name}`

**Description**: Remove identity from gallery

**Parameters**:
- `identity_name` (str, path): Name of identity to delete

**Response**:
```json
{
  "message": "Identity john_doe deleted successfully"
}
```

**cURL Example**:
```bash
curl -X DELETE http://localhost:8000/delete_identity/john_doe
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error description"
}
```

### Common Error Codes

| Code | Error | Cause |
|------|-------|-------|
| 400 | Bad Request | Invalid input or failed processing |
| 404 | Not Found | Identity not found |
| 500 | Internal Server Error | Service error |
| 503 | Service Unavailable | Models not loaded |

### Example Errors

**Invalid Image**:
```json
{
  "detail": "Could not decode image"
}
```

**No Face Detected**:
```json
{
  "detail": "Could not detect face in image"
}
```

**Identity Not Found**:
```json
{
  "detail": "Identity not found"
}
```

---

## Parameters & Configuration

### Recognition Threshold

The `threshold` parameter controls the minimum similarity for face matching.

- **Recommended Range**: 0.55-0.65
- **Lower threshold**: More matches, higher false positives
- **Higher threshold**: Fewer matches, higher false negatives
- **Default**: 0.60

**Guidance**:
- 0.50: Very permissive (many false positives)
- 0.60: Balanced (recommended)
- 0.70: Conservative (fewer false positives)
- 0.80: Very strict (only near-identical faces)

### Image Requirements

- **Format**: JPEG, PNG
- **Resolution**: Minimum 200×200 pixels
- **Face Size**: At least 50×50 pixels recommended
- **Lighting**: Adequate illumination (not extremely dark/bright)
- **Pose**: Face mostly frontal (-45° to +45° angle)

---

## Rate Limiting & Performance

### Performance Expectations

- **Detection Only**: 45 ms (22 FPS)
- **Full Recognition**: 65 ms (15 FPS)
- **Batch Processing**: Linearly scales with batch size

### Concurrent Requests

- **Recommended**: 4-8 concurrent requests
- **Max Workers**: Configured in container
- **Queue Size**: Unlimited

---

## Usage Examples

### Python Client

```python
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

# Recognize faces in image
with open("image.jpg", "rb") as f:
    files = {"file": f}
    data = {"threshold": 0.6}
    response = requests.post(f"{BASE_URL}/recognize", files=files, data=data)
    print(response.json())

# Add identity
files = [
    ("files", open("john_1.jpg", "rb")),
    ("files", open("john_2.jpg", "rb")),
]
data = {"identity_name": "john_doe"}
response = requests.post(f"{BASE_URL}/add_identity", files=files, data=data)
print(response.json())

# Verify faces
files = {
    "file1": open("person_a_1.jpg", "rb"),
    "file2": open("person_a_2.jpg", "rb"),
}
response = requests.post(f"{BASE_URL}/verify", files=files)
print(response.json())
```

### JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const BASE_URL = "http://localhost:8000";

async function recognize(imagePath, threshold = 0.6) {
    const form = new FormData();
    form.append('file', fs.createReadStream(imagePath));
    form.append('threshold', threshold);
    
    const response = await fetch(`${BASE_URL}/recognize`, {
        method: 'POST',
        body: form
    });
    
    return response.json();
}

// Usage
recognize("image.jpg").then(console.log);
```

---

## Postman Collection

Import this JSON into Postman for API testing:

```json
{
  "info": {
    "name": "Face Recognition API",
    "version": "1.0.0"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": "{{base_url}}/health"
      }
    },
    {
      "name": "Detect Faces",
      "request": {
        "method": "POST",
        "url": "{{base_url}}/detect",
        "body": {
          "mode": "formdata",
          "formdata": [
            {"key": "file", "type": "file"}
          ]
        }
      }
    }
  ]
}
```

---

## Deployment

### Docker Deployment

```bash
# Build
docker build -t face-recognition-service .

# Run
docker run -p 8000:8000 \
    -v $(pwd)/database:/app/database \
    -v $(pwd)/models_weights:/app/models_weights \
    face-recognition-service
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: face-recognition-service
  template:
    metadata:
      labels:
        app: face-recognition-service
    spec:
      containers:
      - name: service
        image: face-recognition-service:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## Support

For issues, feature requests, or documentation corrections, please refer to the project repository.
