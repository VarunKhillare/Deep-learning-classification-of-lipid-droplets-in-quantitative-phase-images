# validate.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import list_pairs, split_pairs, build_dataset

# ---------- FIXED CONFIG ----------
RAW_DIR = "/mnt/DATA/EE22B062/MLPE_PROJECT/original_dataset/raw"
MASK_DIR = "/mnt/DATA/EE22B062/MLPE_PROJECT/original_dataset/binary"
MODELS_DIR = "./models"
PATCH = 9
SAMPLES_VAL = 2000
# ----------------------------------

if __name__ == "__main__":
    np.random.seed(42); tf.random.set_seed(42)

    pairs = list_pairs(RAW_DIR, MASK_DIR)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.6, 0.2, 0.2)

    print("Validating on", len(val_pairs), "images...")

    X_val, y_val = build_dataset(val_pairs, patch_size=PATCH, samples_per_image=SAMPLES_VAL)
    y_val = y_val.astype('float32')

    model_names = ["mlp_small", "mlp_med", "mlp_big"]
    results = {}

    for name in model_names:
        path = os.path.join(MODELS_DIR, name + ".h5")
        print("\n=== Evaluating", name, "===")
        model = tf.keras.models.load_model(path, compile=False)

        probs = model.predict(X_val, batch_size=4096, verbose=1).ravel()
        preds = (probs > 0.5).astype(np.float32)

        results[name] = {
            'precision': float(precision_score(y_val, preds, zero_division=0)),
            'recall': float(recall_score(y_val, preds, zero_division=0)),
            'f1': float(f1_score(y_val, preds, zero_division=0))
        }
        print(results[name])

    best = max(results.items(), key=lambda kv: kv[1]['f1'])[0]
    print("\nBEST MODEL:", best, results[best])
