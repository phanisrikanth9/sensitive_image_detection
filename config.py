"""Central configuration for the Visual Privacy Risk Detector.

Every path is relative to this file, so the project runs the same on any machine
(the old version hard-coded C:\\Users\\SRIKANTH\\Desktop\\...). Paths can be
overridden with environment variables.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _path(env_var, default):
    return os.environ.get(env_var, os.path.join(BASE_DIR, default))


class Config:
    # ---------------- Models ----------------
    IMAGE_MODEL_PATH = _path("IMAGE_MODEL_PATH", "models/sensitive_model_final.keras")
    TEXT_MODEL_PATH = _path("TEXT_MODEL_PATH", "models/text_classifier.joblib")

    # Image classifier (MobileNetV2 transfer learning)
    IMAGE_SIZE = (224, 224)
    IMAGE_THRESHOLD = float(os.environ.get("IMAGE_THRESHOLD", 0.5))

    # ---------------- Text dataset ----------------
    TEXT_DATASET_PATH = _path("TEXT_DATASET_PATH", "data/text_dataset.csv")
    SAMPLES_PER_TEMPLATE = 40     # how many filled-in examples per sentence template
    OCR_NOISE_PROB = 0.3          # share of samples that get simulated OCR errors
    RANDOM_STATE = 42

    # ---------------- Text model training ----------------
    # Split is done by *template*, so test sentences use wordings the model never saw.
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    # Missing sensitive content is worse than a false alarm, so the decision
    # threshold is the highest one that still keeps this recall on validation data.
    TARGET_RECALL = 0.95
    METRICS_PATH = _path("METRICS_PATH", "models/text_metrics.json")

    # ---------------- Web app ----------------
    MAX_UPLOAD_MB = 16
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}
    # Return OCR text with PII masked (never the raw values) to the browser.
    RETURN_MASKED_TEXT = True
