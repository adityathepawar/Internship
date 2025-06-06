import cv2
import face_recognition
import numpy as np
import pandas as pd
import os

DATA_DIR = "dataset"
EXCEL_FILE = "face_data.xlsx"

# Save face image uploaded via API
def save_face_image(name, angle, file_bytes):
    filepath = os.path.join(DATA_DIR, f"{name}_{angle}.jpg")
    np_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return "Error reading image."

    # Optional: detect face here and reject if no face found
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        return "No face detected. Please upload a valid image."

    cv2.imwrite(filepath, img)
    return f"Image saved as {filepath}"

# Extract encodings and save to Excel
def extract_encodings():
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Angle", "Encoding"])

    for file in os.listdir(DATA_DIR):
        if file.endswith(".jpg"):
            path = os.path.join(DATA_DIR, file)
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)

            if encodings:
                name, angle = file[:-4].rsplit("_", 1)
                encoding_list = encodings[0].tolist()

                existing = df[(df["Name"] == name) & (df["Angle"] == angle)]
                if not existing.empty:
                    continue  # Skip duplicates
                df.loc[len(df)] = [name, angle, encoding_list]

    df.drop_duplicates(subset=["Name", "Angle"], keep='first', inplace=True)
    df.to_excel(EXCEL_FILE, index=False)
    return "Face encodings extracted and saved."

# Recognize face from uploaded image
def recognize_face_image(file_bytes):
    if not os.path.exists(EXCEL_FILE):
        return "Unknown"

    df = pd.read_excel(EXCEL_FILE)
    if df.empty:
        return "Unknown"

    known_encodings = df['Encoding'].apply(eval).tolist()
    known_names = df['Name'].tolist()

    np_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return "Unknown"

    rgb_small = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        if True in matches:
            first_match_index = matches.index(True)
            return known_names[first_match_index]

    return "Unknown"
