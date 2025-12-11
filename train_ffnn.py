# train_ffnn.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from utils import list_pairs, split_pairs, build_dataset

# -------------------- FIXED CONFIG --------------------
RAW_DIR = "/mnt/DATA/EE22B062/MLPE_PROJECT/original_dataset/raw"
MASK_DIR = "/mnt/DATA/EE22B062/MLPE_PROJECT/original_dataset/binary"
MODELS_DIR = "./models"

PATCH = 9
SAMPLES = 2000

# 🔥 All model configs in one list (clean, compact)
configs = [
    {
        'name': 'mlp_small',
        'hidden': [128, 64],
        'dropout': 0.2,
        'lr': 1e-3,
        'epochs': 10,
        'batch': 1024
    },
    {
        'name': 'mlp_med',
        'hidden': [256, 128, 64],
        'dropout': 0.3,
        'lr': 5e-4,
        'epochs': 12,
        'batch': 2048
    },
    {
        'name': 'mlp_big',
        'hidden': [512, 256, 128],
        'dropout': 0.4,
        'lr': 3e-4,
        'epochs': 14,
        'batch': 4096
    },
]
# ------------------------------------------------------

def build_mlp(input_dim, hidden_layers, dropout, lr):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=x)
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss=losses.BinaryCrossentropy() 
    )
    return model


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    # ---------- Load & split ----------
    pairs = list_pairs(RAW_DIR, MASK_DIR)
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, 0.6, 0.2, 0.2)

    print("Total pairs:", len(pairs))
    print("Train/Val/Test:", len(train_pairs), len(val_pairs), len(test_pairs))

    # ---------- Training dataset ----------
    print("Building TRAIN dataset...")
    X_train, y_train = build_dataset(train_pairs, patch_size=PATCH, samples_per_image=SAMPLES)

    input_dim = PATCH * PATCH

    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---------- Train each model ----------
    for cfg in configs:
        print(f"\n=== Training {cfg['name']} ===")
        model = build_mlp(
            input_dim,
            hidden_layers=cfg['hidden'],
            dropout=cfg['dropout'],
            lr=cfg['lr']
        )

        model.fit(
            X_train, y_train,
            epochs=cfg['epochs'],
            batch_size=cfg['batch'],
            verbose=1
        )

        save_path = os.path.join(MODELS_DIR, cfg['name'] + ".h5")
        model.save(save_path)
        print("Saved:", save_path)

    print("\nDONE — All models trained successfully.")
