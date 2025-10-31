import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Dataset path
DATASET_DIR = r"C:\Users\Dell\OneDrive\Attachments\Data science course\garbage predictor\garbage_classification"

IMG_SIZE = (64, 64)

# Classes
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


X, y = [], []

# Load dataset
for cls in class_to_department.keys():
    folder = os.path.join(DATASET_DIR, cls)
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = gray.flatten()   # Convert image → vector
        X.append(features)
        y.append(cls)

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "garbage_ml_model.pkl")


# Load trained model
model = joblib.load("garbage_ml_model.pkl")




