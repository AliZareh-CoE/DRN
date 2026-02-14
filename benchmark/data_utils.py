"""
Shared data loading utilities for benchmark suite.
Reproduces the exact same train/test split used by DRN in train.py.
"""

import sys
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add parent dir so we can import existing DRN modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_data, create_balanced_odor_groups, normalize_data_robust, enhance_features_advanced
from config import CONFIG


def get_data_paths():
    """Resolve data paths relative to project root."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Data is in /home/user/DRN/no/
    ob_path = os.path.join(project_root, "no", "OB_band_powers.pkl")
    pcx_path = os.path.join(project_root, "no", "PCx_band_powers.pkl")
    meta_path = os.path.join(project_root, "no", "OB_trials_meta.csv")
    return ob_path, pcx_path, meta_path


def load_balanced_data():
    """
    Load data and create balanced 4-class split.
    Returns: X_ob (N,32,21), X_pcx (N,32,21), y_encoded (N,), label_encoder
    """
    ob_path, pcx_path, meta_path = get_data_paths()
    X_ob, X_pcx, y, band_names = load_data(ob_path, pcx_path, meta_path)

    # Handle meta CSV / pickle size mismatch: truncate labels to match arrays
    n_samples = min(X_ob.shape[0], X_pcx.shape[0], len(y))
    if len(y) != X_ob.shape[0]:
        print(f"  Warning: label count ({len(y)}) != array size ({X_ob.shape[0]}), "
              f"truncating to {n_samples}")
        X_ob = X_ob[:n_samples]
        X_pcx = X_pcx[:n_samples]
        y = y[:n_samples]

    np.random.seed(42)
    X_ob_bal, X_pcx_bal, y_groups, y_encoded, y_orig, group_map, label_encoder = \
        create_balanced_odor_groups(X_ob, X_pcx, y, selected_ester=CONFIG['ester_representative'])
    return X_ob_bal, X_pcx_bal, y_encoded, label_encoder


def create_train_test_split(X_ob, X_pcx, y_encoded, seed=42, test_per_class=50):
    """
    Reproduce the EXACT train/test split from train.py lines 132-149.
    This matches train_ensemble() splitting logic.
    """
    num_classes = len(np.unique(y_encoded))
    class_indices = [np.where(y_encoded == i)[0] for i in range(num_classes)]

    np.random.seed(seed)
    test_indices, train_indices = [], []
    for cls in range(num_classes):
        idx = class_indices[cls].copy()
        np.random.shuffle(idx)
        test_indices.extend(idx[:test_per_class])
        train_indices.extend(idx[test_per_class:])

    return (
        X_ob[train_indices], X_pcx[train_indices], y_encoded[train_indices],
        X_ob[test_indices], X_pcx[test_indices], y_encoded[test_indices],
    )


def flatten_for_ml(X_ob_raw, X_pcx_raw):
    """
    Flatten 3D arrays to 2D for classical/AutoML methods.
    Input:  X_ob (N, 32, 21), X_pcx (N, 32, 21)
    Output: X_flat (N, 1344)  -- raw features, no enhancement
    """
    N = X_ob_raw.shape[0]
    ob_flat = X_ob_raw.reshape(N, -1)
    pcx_flat = X_pcx_raw.reshape(N, -1)
    return np.concatenate([ob_flat, pcx_flat], axis=1)


def prepare_all_data():
    """
    Master data preparation function.
    Returns dict with all needed arrays and metadata.
    """
    X_ob, X_pcx, y_encoded, label_encoder = load_balanced_data()

    (X_ob_tr, X_pcx_tr, y_tr,
     X_ob_te, X_pcx_te, y_te) = create_train_test_split(X_ob, X_pcx, y_encoded)

    X_train_flat = flatten_for_ml(X_ob_tr, X_pcx_tr)
    X_test_flat = flatten_for_ml(X_ob_te, X_pcx_te)

    # StandardScaler for classical ML (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    return {
        # Raw 3D
        'X_ob_train': X_ob_tr, 'X_pcx_train': X_pcx_tr, 'y_train': y_tr,
        'X_ob_test': X_ob_te, 'X_pcx_test': X_pcx_te, 'y_test': y_te,
        # Flattened + scaled for ML
        'X_train_flat': X_train_scaled, 'X_test_flat': X_test_scaled,
        # Raw flattened (for methods that do their own scaling)
        'X_train_flat_raw': X_train_flat, 'X_test_flat_raw': X_test_flat,
        # Metadata
        'n_train': len(y_tr), 'n_test': len(y_te),
        'n_features': X_train_flat.shape[1],
        'n_classes': len(np.unique(y_tr)),
        'label_encoder': label_encoder,
    }


if __name__ == '__main__':
    data = prepare_all_data()
    print(f"Train: {data['n_train']} samples, {data['n_features']} features, {data['n_classes']} classes")
    print(f"Test:  {data['n_test']} samples")
    print(f"OB train shape: {data['X_ob_train'].shape}")
    print(f"PCx train shape: {data['X_pcx_train'].shape}")
    print(f"Flat train shape: {data['X_train_flat'].shape}")
