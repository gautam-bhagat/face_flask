from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import face_recognition
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

class ImagePayload(BaseModel):
    image: str  # Base64 encoded image string

class MatchRequest(BaseModel):
    image: str  # Base64 encoded image string
    encodings: list[list[float]]  # List of face encodings

@app.post("/get_face_encoding")
async def get_face_encoding(payload: ImagePayload):
    try:
        # Decode the base64 string
        image_data = base64.b64decode(payload.image)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert the image to RGB (face_recognition requires RGB)
        image = image.convert('RGB')
        
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image_array)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="Could not encode face")
        
        # Convert numpy array to list for JSON serialization
        face_encoding = face_encodings[0].tolist()
        
        return {"face_encoding": face_encoding}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match_face")
async def match_face(request: MatchRequest):
    try:
        # Decode the base64 string
        image_data = base64.b64decode(request.image)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert the image to RGB (face_recognition requires RGB)
        image = image.convert('RGB')
        
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image_array)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image_array, face_locations)
        
        if not face_encodings:
            raise HTTPException(status_code=400, detail="Could not encode face")
        
        # Convert provided encodings to numpy array
        known_encodings = np.array(request.encodings)
        
        # Compare the found face encoding with provided encodings
        matches = face_recognition.compare_faces(known_encodings, face_encodings[0])
        
        # Find the best match
        face_distances = face_recognition.face_distance(known_encodings, face_encodings[0])
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            return {"match_index": int(best_match_index), "distance": float(face_distances[best_match_index])}
        else:
            return {"match_index": -1, "distance": None, "message": "No match found"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/")
async def root():
    return {"message": "Face encoding API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)