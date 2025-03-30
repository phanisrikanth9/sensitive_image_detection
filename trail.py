# trial.py
import sys
import re
import os
import cv2
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from paddleocr import PaddleOCR
from config import Config  # Import your configuration

# Import PatternDetector from your module and inject it into __main__
from pattern_detector import PatternDetector
sys.modules['__main__'].PatternDetector = PatternDetector

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --------------------------
# Image Classification Setup
# --------------------------
model = load_model(r'C:\Users\SRIKANTH\Desktop\sensitive_image_detection\model_training\sensitive_model_final.keras')

# --------------------------
# OCR Setup
# --------------------------
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

# --------------------------
# Text Classification Setup
# --------------------------
text_classifier_dict = joblib.load('text_classification_pipeline.pkl')
text_pipeline = text_classifier_dict['model']
text_threshold = text_classifier_dict.get('threshold', 0.5)
# Text Processing Functions
# --------------------------
def clean_text(text):
    """Normalize text for pattern matching"""
    return ' '.join(text.split())

def contains_sensitive_pattern(text):
    """Check for sensitive patterns with validation"""
    text = clean_text(text)
    for name, pattern in Config.SENSITIVE_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            # Add additional validation checks here if needed
            return True
    return False

# --------------------------
# Image Processing Functions
# --------------------------
def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img).astype('float32') / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print("Image preprocessing error:", e)
        return None

def preprocess_for_ocr(image_path):
    try:
        return cv2.imread(image_path)
    except Exception as e:
        print("OCR preprocessing error:", e)
        return None

# --------------------------
# Classification Functions
# --------------------------
def extract_text(image_path):
    try:
        img = preprocess_for_ocr(image_path)
        if img is None:
            return ""
        ocr_result = ocr_model.ocr(img, cls=True)
        return ' '.join([word_info[1][0] for line in ocr_result for word_info in line])
    except Exception as e:
        print("Text extraction error:", e)
        return ""

def classify_text(text):
    """Handle empty text explicitly"""
    text = text.strip()
    if len(text) < 3:  # Ignore short/no text
        return False  # Native Python bool
    # Rest of your existing logic
    if contains_sensitive_pattern(text):
        return True  # Native Python bool
    try:
        probs = text_pipeline.predict_proba([text])[0]
        return bool(probs[1] >= text_threshold)  # Convert to native bool
    except Exception as e:
        print("Text classification error:", e)
        return False  # Native Python bool

def classify_image(image_path):
    try:
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return "Error"  # Native Python str
        prediction = model.predict(processed_img)
        return "Sensitive" if prediction[0] > 0.5 else "Non-Sensitive"  # Native Python str
    except Exception as e:
        print("Image classification error:", e)
        return "Error"  # Native Python str

# --------------------------
# Flask Routes
# --------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    results = []
    for file in request.files.getlist('files'):
        if file.filename == '':
            continue

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            image_class = classify_image(file_path)
            extracted_text = extract_text(file_path)
            text_sensitive = classify_text(extracted_text)
            
            # Convert numpy.bool_ to native bool
            text_sensitive = bool(text_sensitive)
            is_sensitive = bool(image_class == "Sensitive" or text_sensitive)
            
            results.append({
                "filename": file.filename,
                "sensitive": is_sensitive,
                "image_classification": image_class,
                "text_sensitive": text_sensitive,
                "extracted_text": extracted_text
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return jsonify(results)
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)