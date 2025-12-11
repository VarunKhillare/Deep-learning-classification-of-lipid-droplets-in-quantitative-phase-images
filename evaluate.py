# evaluate.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from skimage.io import imsave
import pandas as pd

from utils import (
    list_pairs,
    split_pairs,
    load_image,
    preprocess_mask,
    predict_full_image_mlp,
    compute_pixel_metrics,
)

# ---------- FIXED CONFIG ----------
RAW_DIR = "/mnt/DATA/EE22B062/MLPE_PROJECT/original_dataset/raw"
MASK_DIR = "/mnt/DATA/EE22B062/MLPE_PROJECT/original_dataset/binary"

MODELS_DIR = "./models"
MODEL_NAME = "mlp_big"          
OUT_DIR = "./predictions_mlp_big"

PATCH = 9
BATCH = 16384
THRESH = 0.5                    # fixed threshold
# ----------------------------------


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # load pairs and split into train/val/test
    pairs = list_pairs(RAW_DIR, MASK_DIR)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, train_frac=0.6, val_frac=0.2, test_frac=0.2
    )
    print("Total pairs:", len(pairs))
    print("Test images:", len(test_pairs))

    # load chosen model
    model_path = os.path.join(MODELS_DIR, MODEL_NAME + ".h5")
    print("Loading model:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    for rawp, maskp in test_pairs:
        raw = load_image(rawp)
        mask = preprocess_mask(load_image(maskp))

        # probability map from MLP
        probs = predict_full_image_mlp(
            model, raw, patch_size=PATCH, batch_size=BATCH
        )
        pred_bin = (probs > THRESH).astype(np.uint8)

        metrics = compute_pixel_metrics(mask, pred_bin)

        base = os.path.splitext(os.path.basename(rawp))[0]
        # save images
        imsave(os.path.join(OUT_DIR, base + "_prob.png"),
               (probs * 255).astype(np.uint8))
        imsave(os.path.join(OUT_DIR, base + "_pred.png"),
               (pred_bin * 255).astype(np.uint8))

        results.append({
            "image": base,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    # save metrics
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "test_metrics.csv"), index=False)
    with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
        import json
        json.dump(results, f, indent=2)

    print("\nSaved predictions and metrics to", OUT_DIR)
    print("Test mean precision:", df["precision"].mean())
    print("Test mean recall   :", df["recall"].mean())
    print("Test mean F1       :", df["f1"].mean())
    print("Test F1 std        :", df["f1"].std())
