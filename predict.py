import cv2
import numpy as np
import joblib

IMG_SIZE = (64, 64)

# Same mapping
class_to_department = {
    "plastic": "Recyclable",
    "plastic soap": "Recyclable",
    "paper": "Recyclable",
    "cardboard": "Recyclable",
    "brown glass": "Recyclable",
    "white glass": "Recyclable",
    "metal": "Recyclable",
    "biological": "Wet/Organic",
    "cloths": "Dry/General",
    "green cloths": "Dry/General",
    "trash": "Dry/General"
}

# Load trained modelgit
model = joblib.load("garbage_ml_model.pkl")

def predict_garbage(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return "❌ Error: image not found"
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = gray.flatten().reshape(1, -1)

    pred_class = model.predict(features)[0]
    department = class_to_department[pred_class]

    return f"Garbage Type: {pred_class}\nDepartment: {department}"

# Example
print(predict_garbage(r"C:\Users\Dell\OneDrive\Attachments\Data science course\garbage predictor\palstic bottel.jpg"))  
