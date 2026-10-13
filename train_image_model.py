"""Train the whole-image sensitive / non-sensitive classifier (MobileNetV2 transfer learning).

This script was missing from the repo. It rebuilds the same architecture found in
sensitive_model_final.keras (read from its saved config):

    224x224x3 input
      -> MobileNetV2 (ImageNet weights, no top)
      -> GlobalAveragePooling2D
      -> Dropout(0.5)
      -> Dense(1, sigmoid)
    loss = binary cross-entropy, optimizer = Adam, metrics = accuracy/precision/recall
    stage 1: backbone frozen, train only the new head
    stage 2: unfreeze the last 6 backbone layers (block_16_* + Conv_1), Adam lr = 1e-5

Changes vs. the original:
  * pixels are scaled to [-1, 1] INSIDE the model (MobileNetV2's expected input);
    the original fed pixel/255 = [0, 1], which does not match ImageNet pre-training
  * data augmentation layers (flip / rotate / zoom / contrast)
  * class weights for imbalanced folders, early stopping, best-checkpoint saving
  * a held-out test folder evaluated once at the end, with a confusion matrix

Expected folder layout (any image names):
    image_data/train/sensitive/*.jpg      image_data/train/non_sensitive/*.jpg
    image_data/val/sensitive/...          image_data/val/non_sensitive/...
    image_data/test/sensitive/...         image_data/test/non_sensitive/...

    python train_image_model.py --data image_data --epochs 10 --fine-tune-epochs 10
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import Config

CLASS_NAMES = ["non_sensitive", "sensitive"]  # index 1 = positive class


def load_split(root, split, batch, shuffle):
    return keras.utils.image_dataset_from_directory(
        os.path.join(root, split), labels="inferred", label_mode="binary",
        class_names=CLASS_NAMES, image_size=Config.IMAGE_SIZE, batch_size=batch,
        shuffle=shuffle, seed=Config.RANDOM_STATE,
    )


def build_model(unfreeze_last=0, weights="imagenet"):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")

    base = keras.applications.MobileNetV2(input_shape=Config.IMAGE_SIZE + (3,),
                                          include_top=False, weights=weights)
    base.trainable = unfreeze_last > 0
    if unfreeze_last:
        for layer in base.layers[:-unfreeze_last]:
            layer.trainable = False

    inputs = keras.Input(shape=Config.IMAGE_SIZE + (3,))
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="rescaling")(x)   # [0,255] -> [-1,1]
    x = base(x, training=False)      # keep BatchNorm statistics frozen while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")])


def class_weights(ds):
    y = np.concatenate([b.numpy().ravel() for _, b in ds])
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return {0: len(y) / (2 * max(n_neg, 1)), 1: len(y) / (2 * max(n_pos, 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="image_data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--unfreeze-last", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=Config.IMAGE_MODEL_PATH)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (smoke tests only)")
    args = ap.parse_args()

    train = load_split(args.data, "train", args.batch, True)
    val = load_split(args.data, "val", args.batch, False)
    weights = class_weights(train)
    train, val = train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_auc", mode="max", save_best_only=True),
    ]

    # Stage 1: train the new classification head on frozen ImageNet features
    model, base = build_model(weights=None if args.weights == "none" else args.weights)
    compile_model(model, 1e-3)
    model.fit(train, validation_data=val, epochs=args.epochs, class_weight=weights, callbacks=callbacks)

    # Stage 2: fine-tune the top of the backbone with a small learning rate
    if args.fine_tune_epochs and args.unfreeze_last:
        base.trainable = True
        for layer in base.layers[:-args.unfreeze_last]:
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(train, validation_data=val, epochs=args.fine_tune_epochs,
                  class_weight=weights, callbacks=callbacks)

    model.save(args.out)

    test_dir = os.path.join(args.data, "test")
    if os.path.isdir(test_dir):
        test = load_split(args.data, "test", args.batch, False)
        y_true = np.concatenate([b.numpy().ravel() for _, b in test]).astype(int)
        probs = model.predict(test, verbose=0).ravel()
        pred = (probs >= Config.IMAGE_THRESHOLD).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum()); tn = int(((pred == 0) & (y_true == 0)).sum())
        metrics = {
            "accuracy": (tp + tn) / len(y_true),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "confusion_matrix [[TN FP] [FN TP]]": [[tn, fp], [fn, tp]],
        }
        print(json.dumps(metrics, indent=2))
        with open(os.path.join(os.path.dirname(args.out), "image_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
