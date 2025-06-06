from fastapi import FastAPI, UploadFile, Form, File
import uvicorn
import pandas as pd
from face_recognition_module import save_face_image, extract_encodings, recognize_face_image
import os

app = FastAPI()

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

@app.get("/")
def root():
    return {"message": "Face Recognition API is running ✅"}

@app.post("/register")
async def register_user(
    name: str = Form(...),
    angle: str = Form(...),
    file: UploadFile = File(...)
):
    file_bytes = await file.read()
    result = save_face_image(name, angle, file_bytes)
    return {"result": result}

@app.post("/extract-encodings")
def extract_encodings_endpoint():
    result = extract_encodings()
    return {"result": result}

@app.post("/verify")
async def verify_user(
    file: UploadFile = File(...)
):
    file_bytes = await file.read()
    name = recognize_face_image(file_bytes)
    if name == "Unknown":
        return {"access": "Denied", "matched_user": None}
    else:
        return {"access": "Granted", "matched_user": name}

# If running locally:
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
