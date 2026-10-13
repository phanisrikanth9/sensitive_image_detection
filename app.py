"""Flask web app for the Visual Privacy Risk Detector (previously trail.py).

Run:  python app.py      then open http://127.0.0.1:5000

Privacy by design: uploaded images are processed in memory and never written to
disk, and the browser only receives OCR text with PII already masked.
"""
import os

from flask import Flask, jsonify, request, send_from_directory
from PIL import UnidentifiedImageError
from werkzeug.utils import secure_filename

from analyzer import PrivacyAnalyzer
from config import BASE_DIR, Config

# The same index.html is used for GitHub Pages (in-browser mode) and here (server mode)
app = Flask(__name__, static_folder="web", static_url_path="/web")
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_MB * 1024 * 1024  # reject huge uploads

analyzer = PrivacyAnalyzer()
for w in analyzer.warnings:
    app.logger.warning(w)


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({
        "image_model": analyzer.image_model is not None,
        "ocr": analyzer.ocr is not None,
        "text_model": analyzer.text_model is not None,
        "warnings": analyzer.warnings,
    })


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    results = []
    for file in files:
        name = secure_filename(file.filename or "")
        if not name:
            continue
        if not allowed(name):
            results.append({"filename": name, "error": "Unsupported file type"})
            continue
        try:
            results.append(analyzer.analyze_image(file.read(), name))
        except UnidentifiedImageError:
            results.append({"filename": name, "error": "Not a valid image"})
        except Exception as e:
            app.logger.exception("analysis failed for %s", name)
            results.append({"filename": name, "error": f"Analysis failed: {type(e).__name__}"})
    return jsonify(results)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"Upload larger than {Config.MAX_UPLOAD_MB} MB"}), 413


if __name__ == "__main__":
    # debug=True exposes an interactive debugger to anyone who can reach the port,
    # so it is opt-in via FLASK_DEBUG=1
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
