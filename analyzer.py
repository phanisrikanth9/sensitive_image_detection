"""The full image analysis pipeline, independent of the web framework.

    image ──► MobileNetV2 classifier ──────────────► image score
      │
      └──► PaddleOCR ──► text lines + boxes ──► rule-based PII detector ──► findings (+ boxes to blur)
                                   │
                                   └──► ML text classifier ──► text score

    decision = PII found  OR  text score >= t_text  OR  image score >= t_img

Each component is optional: if TensorFlow / PaddleOCR / a model file is missing,
the analyzer still runs with whatever is available and says so in `warnings`.
"""
import base64
import io
import os

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from config import Config
from detectors import find_pii, mask_text


# ---------------------------------------------------------------------------
# Image classifier
# ---------------------------------------------------------------------------
class ImageClassifier:
    def __init__(self, path=Config.IMAGE_MODEL_PATH):
        from tensorflow.keras.models import load_model  # imported lazily: heavy
        self.model = load_model(path)
        # Models built by train_image_model.py scale pixels inside the network
        # (a Rescaling layer). The original model expects pixels already / 255.
        self.scales_inside = any("rescaling" in l.name for l in self.model.layers)

    def score(self, rgb: Image.Image) -> float:
        x = np.asarray(rgb.resize(Config.IMAGE_SIZE), dtype="float32")[None, ...]
        if not self.scales_inside:
            x = x / 255.0
        return float(self.model.predict(x, verbose=0).ravel()[0])


# ---------------------------------------------------------------------------
# OCR (works with PaddleOCR 2.x and 3.x, whose APIs differ)
# ---------------------------------------------------------------------------
class OCREngine:
    def __init__(self):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import paddleocr
        from paddleocr import PaddleOCR
        self.v3 = int(paddleocr.__version__.split(".")[0]) >= 3
        if self.v3:
            self.engine = PaddleOCR(lang="en", use_textline_orientation=True,
                                    use_doc_orientation_classify=False, use_doc_unwarping=False)
        else:
            self.engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def read(self, bgr: np.ndarray):
        """Return [(text, (x1, y1, x2, y2), confidence), ...] in reading order."""
        lines = []
        if self.v3:
            for res in self.engine.predict(bgr):
                polys = res.get("rec_polys", res.get("dt_polys", []))
                for text, score, poly in zip(res["rec_texts"], res["rec_scores"], polys):
                    lines.append((text, _bbox(poly), float(score)))
        else:
            result = self.engine.ocr(bgr, cls=True) or []
            for page in result:
                for poly, (text, score) in page or []:  # page is None when no text
                    lines.append((text, _bbox(poly), float(score)))
        lines.sort(key=lambda l: (round(l[1][1] / 15), l[1][0]))  # top-to-bottom, left-to-right
        return lines


def _bbox(poly):
    pts = np.asarray(poly, dtype=float).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


# ---------------------------------------------------------------------------
# Text classifier
# ---------------------------------------------------------------------------
class TextClassifier:
    def __init__(self, path=Config.TEXT_MODEL_PATH):
        import joblib
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.threshold = float(bundle.get("threshold", 0.5))

    def score(self, text: str) -> float:
        return float(self.model.predict_proba([text])[0, 1])


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
def redact(rgb: Image.Image, lines, findings_per_line) -> Image.Image:
    """Cover the part of each OCR line that holds PII with a solid black box.

    OCR gives one box per line, so the horizontal position of the PII inside the
    box is estimated from its character offsets, padded by one character width on
    each side. A solid box is used instead of blur/pixelation because pixelated
    text can sometimes be reconstructed ("depixelation" attacks).
    """
    out = rgb.copy()
    draw = ImageDraw.Draw(out)
    for (text, (x1, y1, x2, y2), _), findings in zip(lines, findings_per_line):
        if not findings:
            continue
        char_w = (x2 - x1) / max(len(text), 1)
        for f in findings:
            fx1 = x1 + char_w * f.start - char_w
            fx2 = x1 + char_w * f.end + 1.5 * char_w  # digits are wider than average
            draw.rectangle([max(int(fx1), x1 - 2), y1 - 3, min(int(fx2), x2 + 2), y2 + 3], fill="black")
    return out


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class PrivacyAnalyzer:
    def __init__(self, load_image_model=True, load_ocr=True, load_text_model=True):
        self.warnings = []
        # PaddleOCR must be loaded BEFORE TensorFlow: with recent versions,
        # importing tensorflow first and then paddle crashes with a segfault.
        self.ocr = self._try(load_ocr, OCREngine, "OCR")
        self.image_model = self._try(load_image_model, ImageClassifier, "image classifier")
        self.text_model = self._try(load_text_model, TextClassifier, "text classifier")

    def _try(self, enabled, factory, label):
        if not enabled:
            return None
        try:
            return factory()
        except Exception as e:  # missing package or model file
            self.warnings.append(f"{label} unavailable: {type(e).__name__}: {e}")
            return None

    def analyze_text(self, text: str):
        findings = find_pii(text)
        text_score = self.text_model.score(text) if (self.text_model and len(text.strip()) >= 3) else None
        return findings, text_score

    def analyze_image(self, file_bytes: bytes, filename: str = "", redact_output: bool = True):
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
        result = {"filename": filename, "reasons": []}

        # 1) whole-image classifier
        image_score = self.image_model.score(img) if self.image_model else None
        image_flag = image_score is not None and image_score >= Config.IMAGE_THRESHOLD
        result["image"] = {"score": _r(image_score), "threshold": Config.IMAGE_THRESHOLD, "sensitive": image_flag}
        if image_flag:
            result["reasons"].append(f"Image classifier: {image_score:.0%} likely sensitive")

        # 2) OCR -> 3) PII rules per line (keeps positions for redaction)
        lines = self.ocr.read(np.asarray(img)[:, :, ::-1]) if self.ocr else []
        per_line = [find_pii(t) for t, _, _ in lines]
        full_text = " ".join(t for t, _, _ in lines)
        findings = [f for fl in per_line for f in fl]
        # PII split across two OCR lines is only visible in the joined text
        line_values = {f.value for f in findings}
        findings += [f for f in find_pii(full_text) if f.value not in line_values]
        if findings:
            types = sorted({f.type for f in findings})
            result["reasons"].append("Personal data found: " + ", ".join(types))

        # 4) ML text classifier for context (passwords, medical, "confidential"...)
        _, text_score = self.analyze_text(full_text) if full_text else ([], None)
        t_thr = self.text_model.threshold if self.text_model else None
        text_flag = text_score is not None and text_score >= t_thr
        if text_flag:
            result["reasons"].append(f"Text classifier: {text_score:.0%} likely sensitive content")

        result["text"] = {"score": _r(text_score), "threshold": t_thr, "sensitive": text_flag}
        result["pii"] = [f.to_dict() for f in findings]
        result["extracted_text"] = mask_text(full_text) if Config.RETURN_MASKED_TEXT else full_text
        result["sensitive"] = bool(image_flag or text_flag or findings)
        result["risk_level"] = "high" if findings else ("medium" if result["sensitive"] else "low")
        if redact_output and any(per_line):
            result["redacted_image"] = to_data_url(redact(img, lines, per_line))

        # keys kept for the original front-end
        result["image_classification"] = ("Sensitive" if image_flag else "Non-Sensitive") if image_score is not None else "Unavailable"
        result["text_sensitive"] = bool(text_flag or findings)
        result["trigger_text"] = ", ".join(f"{f.type}: {f.masked}" for f in findings)
        return result


def _r(x):
    return None if x is None else round(float(x), 4)
