# utils.py
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

# ---------- I/O & pairing ----------

def list_pairs(raw_dir, mask_dir):
    """Return list of (raw_path, mask_path) pairs matched by basename."""
    raw_files = sorted([f for f in os.listdir(raw_dir) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])
    baseraw = {os.path.splitext(f)[0]: f for f in raw_files}
    basemask = {os.path.splitext(f)[0]: f for f in mask_files}
    common = sorted(list(set(baseraw.keys()).intersection(basemask.keys())))
    pairs = [(os.path.join(raw_dir, baseraw[k]), os.path.join(mask_dir, basemask[k])) for k in common]
    return pairs

# ---------- image load / normalize ----------

def load_image(path):
    """Load grayscale image (float or uint)."""
    im = imread(path)
    if im.ndim == 3:
        im = im[..., 0]
    return im

def normalize_raw(img):
    """Scale raw image to [0,1]. Handles uint and float."""
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        return rescale_intensity(img, in_range='image', out_range=(0, 1)).astype(np.float32)
    else:
        lo, hi = np.percentile(img, (1, 99))
        imgc = np.clip(img, lo, hi)
        if imgc.max() - imgc.min() < 1e-9:
            return np.zeros_like(imgc, dtype=np.float32)
        return ((imgc - imgc.min()) / (imgc.max() - imgc.min())).astype(np.float32)

def preprocess_mask(mask):
    """Ensure binary 0/1 mask."""
    u = np.unique(mask)
    if set(u).issubset({0, 1}):
        return mask.astype(np.uint8)
    return (mask > (mask.max() / 2)).astype(np.uint8)

# ---------- split pairs (image-level 60/20/20) ----------

def split_pairs(pairs, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=SEED):
    """Split list of pairs into train/val/test at image level."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    idx = np.arange(len(pairs))
    train_idx, rest_idx = train_test_split(idx, test_size=(1 - train_frac), random_state=seed)
    val_idx, test_idx = train_test_split(rest_idx, test_size=(test_frac / (test_frac + val_frac)),
                                         random_state=seed)
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    return train_pairs, val_pairs, test_pairs

# ---------- patch extraction ----------

def extract_patches_from_image(raw, mask, patch_size=9, sample_count=2000):
    """
    Extract random per-pixel patches and labels from a single image.
    raw, mask: 2D arrays, already normalized/binary.
    """
    H, W = raw.shape
    half = patch_size // 2
    # pad for border handling
    rp = np.pad(raw, half, mode='constant', constant_values=0)
    mp = np.pad(mask, half, mode='constant', constant_values=0)

    coords = [(y, x) for y in range(half, half + H) for x in range(half, half + W)]
    if sample_count is not None and sample_count < len(coords):
        idx = np.random.choice(len(coords), size=sample_count, replace=False)
        coords = [coords[i] for i in idx]

    X = np.zeros((len(coords), patch_size * patch_size), dtype=np.float32)
    y = np.zeros((len(coords),), dtype=np.float32)

    for i, (yy, xx) in enumerate(coords):
        patch = rp[yy - half:yy + half + 1, xx - half:xx + half + 1]
        X[i] = patch.reshape(-1)
        y[i] = float(mp[yy, xx])

    return X, y

def build_dataset(pairs, patch_size=9, samples_per_image=2000, show_progress=True):
    """Build feature matrix X and labels y from a list of image pairs."""
    Xlist = []
    ylist = []
    if show_progress:
        from tqdm import tqdm
        pairs_iter = tqdm(pairs)
    else:
        pairs_iter = pairs

    for rawp, maskp in pairs_iter:
        r = normalize_raw(load_image(rawp))
        m = preprocess_mask(load_image(maskp))
        X, y = extract_patches_from_image(r, m, patch_size, samples_per_image)
        Xlist.append(X)
        ylist.append(y)

    X = np.vstack(Xlist)
    y = np.concatenate(ylist)  # already float32
    return X, y

# ---------- full-image prediction for MLP ----------

def predict_full_image_mlp(model, raw, patch_size=9, batch_size=16384):
    """
    Run per-pixel patch-based prediction on a full raw image using MLP model.
    Returns probability map (float32).
    """
    raw = normalize_raw(raw)
    H, W = raw.shape
    half = patch_size // 2
    rp = np.pad(raw, half, mode='constant', constant_values=0)

    coords = [(y, x) for y in range(half, half + H) for x in range(half, half + W)]
    preds = np.zeros((H * W,), dtype=np.float32)
    batch = []
    cnt = 0

    for i, (yy, xx) in enumerate(coords):
        patch = rp[yy - half:yy + half + 1, xx - half:xx + half + 1].reshape(-1)
        batch.append(patch)
        if len(batch) >= batch_size or i == len(coords) - 1:
            Xb = np.array(batch, dtype=np.float32)
            p = model.predict(Xb, batch_size=4096, verbose=0).ravel()
            preds[cnt:cnt + len(p)] = p
            cnt += len(p)
            batch = []

    return preds.reshape(H, W)

# ---------- metrics ----------

def compute_pixel_metrics(y_true, y_pred_bin):
    """
    Compute per-pixel precision, recall, F1.
    y_true, y_pred_bin: 2D arrays (0/1).
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true = y_true.flatten()
    y_pred = y_pred_bin.flatten()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
