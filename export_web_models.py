"""Export the trained models so the browser version (index.html) can run them.

    python export_web_models.py

Writes:
  web/text_model.json           TF-IDF vocabularies + IDF weights + logistic regression
  web/image_model/graph.json    MobileNetV2 layer graph (from the .keras config)
  web/image_model/weights.bin   all weights, float32, concatenated
"""
import json
import os
import zipfile

import joblib
import numpy as np

from config import BASE_DIR, Config
from pattern_detector import FEATURE_NAMES, KEYWORD_GROUPS

OUT = os.path.join(BASE_DIR, "web")


def export_text_model():
    bundle = joblib.load(Config.TEXT_MODEL_PATH)
    pipe = bundle["model"]
    union = pipe.named_steps["features"]
    clf = pipe.named_steps["clf"]
    vecs = dict(union.transformer_list)
    coef = clf.coef_.ravel()

    def vec_json(v, offset):
        vocab = v.vocabulary_
        n = len(vocab)
        # term -> [idf, coef]; the browser only needs the product structure
        terms = {t: [round(float(v.idf_[i]), 6), round(float(coef[offset + i]), 6)] for t, i in vocab.items()}
        return {"analyzer": v.analyzer, "ngram_range": list(v.ngram_range),
                "sublinear_tf": v.sublinear_tf, "terms": terms}, offset + n

    words, off = vec_json(vecs["words"], 0)
    chars, off = vec_json(vecs["chars"], off)
    pattern_coef = [round(float(c), 6) for c in coef[off:]]
    assert len(pattern_coef) == len(FEATURE_NAMES)
    model = {
        "words": words, "chars": chars,
        "pattern_features": FEATURE_NAMES, "pattern_coef": pattern_coef,
        "keyword_groups": KEYWORD_GROUPS,
        "intercept": float(clf.intercept_[0]), "threshold": float(bundle["threshold"]),
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "text_model.json")
    with open(path, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"text model -> {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def export_image_model():
    """Flatten the Keras functional graph and dump weights in layer order."""
    from tensorflow import keras

    model = keras.models.load_model(Config.IMAGE_MODEL_PATH)
    cfg = json.loads(zipfile.ZipFile(Config.IMAGE_MODEL_PATH).read("config.json"))

    layers, blobs, offset = [], [], 0

    def add(layer_cfg, keras_layer, prefix_inputs):
        nonlocal offset
        c = layer_cfg["config"]
        entry = {"name": c["name"], "type": layer_cfg["class_name"], "inputs": prefix_inputs}
        for k in ("strides", "padding", "activation", "max_value", "epsilon", "kernel_size", "use_bias"):
            if k in c:
                entry[k] = c[k]
        w = []
        for arr in keras_layer.get_weights():
            arr = np.asarray(arr, dtype="<f4")
            w.append({"shape": list(arr.shape), "offset": offset})
            blobs.append(arr.tobytes())
            offset += arr.size
        entry["weights"] = w
        layers.append(entry)

    def inbound(layer_cfg):
        names = []
        for node in layer_cfg.get("inbound_nodes", []):
            for a in node["args"]:
                items = a if isinstance(a, list) else [a]
                for it in items:
                    if isinstance(it, dict) and "config" in it:
                        names.append(it["config"]["keras_history"][0])
        return names

    base = model.get_layer("mobilenetv2_1.00_224")
    for lc in cfg["config"]["layers"][1]["config"]["layers"]:
        if lc["class_name"] == "InputLayer":
            continue
        add(lc, base.get_layer(lc["config"]["name"]), inbound(lc))
    # top: GAP -> Dropout (identity at inference) -> Dense
    last = layers[-1]["name"]
    for lc in cfg["config"]["layers"][2:]:
        if lc["class_name"] == "Dropout":
            continue
        add(lc, model.get_layer(lc["config"]["name"]), [last])
        last = lc["config"]["name"]

    d = os.path.join(OUT, "image_model")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "weights.bin"), "wb") as f:
        for b in blobs:
            f.write(b)
    with open(os.path.join(d, "graph.json"), "w") as f:
        json.dump({"input": [224, 224, 3], "scale": "0_1", "layers": layers}, f, separators=(",", ":"))
    print(f"image model -> {d} ({offset * 4 / 1e6:.1f} MB weights, {len(layers)} layers)")


if __name__ == "__main__":
    export_text_model()
    if os.path.exists(Config.IMAGE_MODEL_PATH):
        export_image_model()
    else:
        print("image model file not found; skipped")
