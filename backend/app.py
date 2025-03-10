import sys
import os

# Add the "src" folder (which contains combined_features.py) to the Python path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib
from PIL import Image
import numpy as np

# Initializing Flask App
app = Flask(__name__)
CORS(app)

# Loading ML Model
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
MODEL_PATH = os.path.join(BASE_DIR, "final_age_detection_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
app.logger.info(f"Loaded label mapping: {label_mapping}")

# Directory for Temporary Uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return "<h1>Welcome to the Age Detection ML API</h1><p>Use the /predict endpoint to get predictions.</p>"


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({'Error': 'No file part in the request'}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({'Error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.info(f"Saving uploaded file to: {file_path}")
    file.save(file_path)
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
        
        app.logger.info(f"File exists, proceeding with prediction: {file_path}")
        
        prediction = model.predict([file_path])
        predicted_int = int(prediction[0])
        predicted_age_group = label_mapping.get(predicted_int, "Unknown")
        result = predicted_age_group
    except Exception as e:
        result = f"Error during prediction: {str(e)}"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            app.logger.info(f"Removed temporary file: {file_path}")
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)