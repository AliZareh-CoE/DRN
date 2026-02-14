"""
Configuration for Architectural Ablation Study.
"""

# =====================================================
# DATA PATHS - Update these for your system
# =====================================================
OB_PKL_PATH = "data/OB_band_powers.pkl"
PCX_PKL_PATH = "data/PCx_band_powers.pkl"
META_CSV_PATH = "data/OB_trials_meta.csv"

# =====================================================
# TRAINING CONFIGURATION - Paper-Consistent Settings
# =====================================================
CONFIG = {
    'ester_representative': 'isoamyl acetate at 0.3% v/v',
    'dropout_rate': 0.25,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'num_epochs': 70,
    'use_early_stopping': False,
    'use_mixup': True,
    'num_folds': 5,
    'use_ensemble': True,
    'extended': False,
}

# =====================================================
# ABLATION STUDY SETTINGS
# =====================================================
NUM_RUNS_PER_EXPERIMENT = 5
RUN_SEEDS = [42, 1337, 2024, 7777, 9999]
NUM_GPUS = 8
OUTPUT_DIR = 'results'

# =====================================================
# ABLATION EXPERIMENT CONFIGURATIONS
# =====================================================
ARCHITECTURAL_ABLATION_CONFIGS = [
    ("Full Model (Baseline)", {}),
    ("-CA (No Channel Attention)", {"use_channel_attention": False}),
    ("-SA (No Spatial Attention)", {"use_spatial_attention": False}),
    ("-FA (No Fusion Attention)", {"use_fusion_attention": False}),
    ("-Skip (No Skip Classifier)", {"use_skip_classifier": False}),
    ("-Residual (No Skip Connections)", {"use_residual_connections": False}),
    ("-CA-SA (No Attention)", {"use_channel_attention": False, "use_spatial_attention": False}),
    ("Activation: ReLU", {"activation": "relu"}),
    ("Activation: GELU", {"activation": "gelu"}),
    ("Activation: ELU", {"activation": "elu"}),
    ("Activation: SiLU", {"activation": "silu"}),
]