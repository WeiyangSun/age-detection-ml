import sys
import os

# Add the "src" folder (which contains combined_features.py) to the Python path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import joblib
from PIL import Image
import numpy as np

# Initializing Flask App
app = Flask(__name__)

# Loading ML Model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "final_age_detection_model.pkl")
model = joblib.load(MODEL_PATH)

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
    file.save(file_path)
    
    try:
        prediction = model.predict([file_path])
        result = prediction[0]
    except Exception as e:
        result = f"Error during prediction: {str(e)}"
    finally:
        os.remove(file_path)
    
    return jsonify({"ML Prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)