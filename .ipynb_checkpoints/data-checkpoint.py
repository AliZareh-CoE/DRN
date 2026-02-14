"""
Data loading and preprocessing for neural odor classification.
"""

import pickle
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_data(ob_pkl_path, pcx_pkl_path, meta_csv_path, selected_bands=None,
              ext_ob_pkl_path=None, ext_pcx_pkl_path=None, ext_meta_csv_path=None):
    """Load OB/PCx band-power pickles and metadata CSV."""
    
    def _load_pkl(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle different pickle structures
        if isinstance(data, dict):
            # Try different key names for the array
            if 'band_powers' in data:
                arr = data['band_powers']
            elif 'data' in data:
                arr = data['data']
            elif 'X' in data:
                arr = data['X']
            else:
                print(f"  Available keys in pickle: {list(data.keys())}")
                raise KeyError(f"Cannot find data array. Keys: {list(data.keys())}")
            
            # Try different key names for bands
            if 'bands' in data:
                bands = data['bands']
            elif 'band_names' in data:
                bands = data['band_names']
            elif 'features' in data:
                bands = data['features']
            else:
                bands = [f'band_{i}' for i in range(arr.shape[-1])]
                print(f"  Warning: No band names found, using generic names")
        elif isinstance(data, np.ndarray):
            arr = data
            bands = [f'band_{i}' for i in range(arr.shape[-1])]
            print(f"  Warning: Pickle is raw array, using generic band names")
        else:
            raise TypeError(f"Unexpected pickle type: {type(data)}")
        
        if isinstance(bands, np.ndarray):
            bands = bands.tolist()
        if len(bands) > 0 and isinstance(bands[0], tuple):
            bands = [b[0] for b in bands]
        
        return arr, bands

    print("Loading primary band-power pickles...")
    X_ob, bands_ob = _load_pkl(ob_pkl_path)
    X_pcx, bands_pcx = _load_pkl(pcx_pkl_path)

    print("Loading metadata CSV...")
    meta = pd.read_csv(meta_csv_path)
    
    # Try different column names for odor labels
    if 'odor_name' in meta.columns:
        y = meta['odor_name'].values
    elif 'odor' in meta.columns:
        y = meta['odor'].values
    elif 'label' in meta.columns:
        y = meta['label'].values
    else:
        print(f"  Available columns: {list(meta.columns)}")
        raise KeyError(f"Cannot find odor column. Columns: {list(meta.columns)}")

    if ext_ob_pkl_path and ext_pcx_pkl_path and ext_meta_csv_path:
        print("Loading extended dataset...")
        X_ob_ext, _ = _load_pkl(ext_ob_pkl_path)
        X_pcx_ext, _ = _load_pkl(ext_pcx_pkl_path)
        meta_ext = pd.read_csv(ext_meta_csv_path)
        if 'odor_name' in meta_ext.columns:
            y_ext = meta_ext['odor_name'].values
        elif 'odor' in meta_ext.columns:
            y_ext = meta_ext['odor'].values
        else:
            y_ext = meta_ext['label'].values
        X_ob = np.concatenate([X_ob, X_ob_ext], axis=0)
        X_pcx = np.concatenate([X_pcx, X_pcx_ext], axis=0)
        y = np.concatenate([y, y_ext], axis=0)

    band_names = bands_ob
    if selected_bands is not None:
        idxs = [band_names.index(b) for b in selected_bands if b in band_names]
        X_ob = X_ob[:, :, idxs]
        X_pcx = X_pcx[:, :, idxs]
        band_names = [band_names[i] for i in idxs]

    print(f"OB shape:   {X_ob.shape}")
    print(f"PCx shape:  {X_pcx.shape}")
    print(f"Samples:    {len(y)}")
    print(f"Bands used: {band_names}")
    print(f"Distribution: {Counter(y)}")

    return X_ob, X_pcx, y, band_names


def create_balanced_odor_groups(X_ob, X_pcx, y, selected_ester='isoamyl acetate at 0.3% v/v'):
    """Create balanced 4-class odor groups."""
    
    group_map = {
        'mineral oil': 'Control',
        'hexanal at 0.3% v/v': 'Aldehyde',
        '2-hexanone at 0.3% v/v': 'Ketone',
    }
    
    ester_odors = [
        'ethyl acetate at 0.3% v/v', 'ethyl butyrate at 0.3% v/v',
        'ethyl tiglate at 0.3% v/v', 'isoamyl acetate at 0.3% v/v',
    ]
    
    for ester in ester_odors:
        if ester == selected_ester:
            group_map[ester] = 'Ester'

    mask = np.array([yi in group_map for yi in y])
    X_ob_filt = X_ob[mask]
    X_pcx_filt = X_pcx[mask]
    y_filt = y[mask]
    y_groups = np.array([group_map[yi] for yi in y_filt])

    unique_groups, counts = np.unique(y_groups, return_counts=True)
    min_count = min(counts)
    
    balanced_idx = []
    for grp in unique_groups:
        grp_idx = np.where(y_groups == grp)[0]
        np.random.shuffle(grp_idx)
        balanced_idx.extend(grp_idx[:min_count])
    
    balanced_idx = np.array(balanced_idx)
    np.random.shuffle(balanced_idx)

    X_ob_bal = X_ob_filt[balanced_idx]
    X_pcx_bal = X_pcx_filt[balanced_idx]
    y_groups_bal = y_groups[balanced_idx]
    y_orig = y_filt[balanced_idx]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_groups_bal)

    print(f"\nBalanced 4-class dataset:")
    print(f"  Selected ester: {selected_ester}")
    print(f"  Samples per class: {min_count}")
    print(f"  Total: {len(y_encoded)}")
    print(f"  Classes: {label_encoder.classes_}")

    return X_ob_bal, X_pcx_bal, y_groups_bal, y_encoded, y_orig, group_map, label_encoder


def normalize_data_robust(X_train, X_val=None, X_test=None):
    """Robust normalization using median and IQR."""
    median = np.median(X_train, axis=0, keepdims=True)
    q75 = np.percentile(X_train, 75, axis=0, keepdims=True)
    q25 = np.percentile(X_train, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0

    X_train_norm = (X_train - median) / iqr
    X_val_norm = (X_val - median) / iqr if X_val is not None else None
    X_test_norm = (X_test - median) / iqr if X_test is not None else None

    return X_train_norm, X_val_norm, X_test_norm


def enhance_features_advanced(X):
    """Advanced feature enhancement with cross-channel/band statistics."""
    n_samples, n_channels, n_bands = X.shape

    chan_mean = np.mean(X, axis=1, keepdims=True)
    chan_std = np.std(X, axis=1, keepdims=True) + 1e-8
    band_mean = np.mean(X, axis=2, keepdims=True)
    band_std = np.std(X, axis=2, keepdims=True) + 1e-8

    chan_mean_exp = np.tile(chan_mean, (1, n_channels, 1))
    chan_std_exp = np.tile(chan_std, (1, n_channels, 1))
    band_mean_exp = np.tile(band_mean, (1, 1, n_bands))
    band_std_exp = np.tile(band_std, (1, 1, n_bands))

    return np.concatenate([X, chan_mean_exp, chan_std_exp, band_mean_exp, band_std_exp], axis=1)