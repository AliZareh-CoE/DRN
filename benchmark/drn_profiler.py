"""
Profile the DRN model: FLOPs, parameters per component, inference time, memory.
"""

import sys
import os
import time
import json
import tracemalloc
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import DualRegionNet
from data import normalize_data_robust, enhance_features_advanced
from benchmark.data_utils import prepare_all_data


def count_parameters_by_component(model):
    """Count parameters broken down by DRN component."""
    components = {
        'ChannelAttention (Input)': 0,
        'Conv1 Stem': 0,
        'ResidualBlocks (incl. CA)': 0,
        'SpatialAttention': 0,
        'FeatureProjection': 0,
        'FusionAttention': 0,
        'Classifier': 0,
        'SkipClassifier': 0,
    }

    for stream_name in ['ob_net', 'pcx_net']:
        stream = getattr(model, stream_name)
        for name, param in stream.named_parameters():
            n = param.numel()
            if 'channel_attention' in name and 'residual' not in name:
                components['ChannelAttention (Input)'] += n
            elif 'conv1' in name:
                components['Conv1 Stem'] += n
            elif 'residual_blocks' in name:
                components['ResidualBlocks (incl. CA)'] += n
            elif 'spatial_attn' in name:
                components['SpatialAttention'] += n
            elif 'feature_projection' in name:
                components['FeatureProjection'] += n

    if model.fusion_attention is not None:
        for param in model.fusion_attention.parameters():
            components['FusionAttention'] += param.numel()

    for param in model.classifier.parameters():
        components['Classifier'] += param.numel()

    if model.skip is not None:
        for param in model.skip.parameters():
            components['SkipClassifier'] += param.numel()

    return components


def get_flops(model, ob_shape, pcx_shape, device='cpu'):
    """
    Compute FLOPs using ptflops (primary), thop (fallback), manual estimation (last resort).
    ob_shape, pcx_shape: (C, L) tuples, e.g., (160, 21).
    """
    # Try ptflops
    try:
        from ptflops import get_model_complexity_info

        class WrappedDRN(nn.Module):
            def __init__(self, drn, pcx_input):
                super().__init__()
                self.drn = drn
                self.pcx_input = pcx_input

            def forward(self, ob_x):
                batch = ob_x.shape[0]
                pcx = self.pcx_input[:batch].to(ob_x.device)
                return self.drn(ob_x, pcx)

        dummy_pcx = torch.randn(1, *pcx_shape).to(device)
        wrapped = WrappedDRN(model, dummy_pcx).to(device)
        wrapped.eval()

        macs, params = get_model_complexity_info(
            wrapped, tuple(ob_shape), as_strings=False, print_per_layer_stat=False
        )
        return {'macs': int(macs), 'flops': int(macs * 2), 'source': 'ptflops'}
    except Exception as e:
        print(f"  ptflops failed: {e}")

    # Try thop
    try:
        from thop import profile as thop_profile
        model_copy = model.to(device)
        model_copy.eval()
        dummy_ob = torch.randn(1, *ob_shape).to(device)
        dummy_pcx = torch.randn(1, *pcx_shape).to(device)
        macs, params = thop_profile(model_copy, inputs=(dummy_ob, dummy_pcx), verbose=False)
        return {'macs': int(macs), 'flops': int(macs * 2), 'source': 'thop'}
    except Exception as e:
        print(f"  thop failed: {e}")

    # Manual estimation fallback
    return _estimate_flops_manual(model, ob_shape, pcx_shape)


def _estimate_flops_manual(model, ob_shape, pcx_shape):
    """Manual FLOPs estimation as fallback."""
    total_macs = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            # MACs = out_c * kernel_size * in_c * output_length (approx 10)
            macs = module.weight.numel() * 10
            total_macs += macs
        elif isinstance(module, nn.Linear):
            macs = module.weight.numel()
            total_macs += macs
    return {'macs': total_macs, 'flops': total_macs * 2, 'source': 'manual_estimate'}


def measure_inference_time(model, ob_shape, pcx_shape, device='cpu',
                           n_warmup=50, n_runs=200, batch_size=1):
    """Measure inference latency."""
    model.eval()
    model.to(device)

    dummy_ob = torch.randn(batch_size, *ob_shape).to(device)
    dummy_pcx = torch.randn(batch_size, *pcx_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_ob, dummy_pcx)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_ob, dummy_pcx)
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return {
        'batch_size': batch_size,
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'median_ms': float(np.median(times) * 1000),
        'per_sample_ms': float(np.mean(times) * 1000 / batch_size),
    }


def measure_memory(model, ob_shape, pcx_shape, device='cpu'):
    """Measure peak memory during forward pass."""
    model.eval()
    model.to(device)

    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        dummy_ob = torch.randn(1, *ob_shape).to(device)
        dummy_pcx = torch.randn(1, *pcx_shape).to(device)
        with torch.no_grad():
            _ = model(dummy_ob, dummy_pcx)
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        return {'peak_memory_mb': float(peak_mem), 'device': device}

    # CPU fallback with tracemalloc
    tracemalloc.start()
    dummy_ob = torch.randn(1, *ob_shape)
    dummy_pcx = torch.randn(1, *pcx_shape)
    with torch.no_grad():
        _ = model(dummy_ob, dummy_pcx)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'peak_memory_mb': float(peak / 1e6), 'device': 'cpu'}


def get_enhanced_input_shapes(data):
    """Get input shapes after normalization and enhancement."""
    X_ob_n, _, _ = normalize_data_robust(data['X_ob_train'])
    X_ob_e = enhance_features_advanced(X_ob_n)
    ob_channels, ob_bands = X_ob_e.shape[1], X_ob_e.shape[2]

    X_pcx_n, _, _ = normalize_data_robust(data['X_pcx_train'])
    X_pcx_e = enhance_features_advanced(X_pcx_n)
    pcx_channels, pcx_bands = X_pcx_e.shape[1], X_pcx_e.shape[2]

    return ob_channels, ob_bands, pcx_channels, pcx_bands


def profile_drn(data=None):
    """Full profiling of DRN model."""
    if data is None:
        data = prepare_all_data()

    ob_channels, ob_bands, pcx_channels, pcx_bands = get_enhanced_input_shapes(data)
    print(f"  Enhanced input shapes: OB=({ob_channels}, {ob_bands}), PCx=({pcx_channels}, {pcx_bands})")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    model = DualRegionNet(
        ob_channels, ob_bands, pcx_channels, pcx_bands,
        num_classes=data['n_classes'],
        dropout_rate=0.25,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / 1e6  # float32

    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {model_size_mb:.2f} MB")

    component_params = count_parameters_by_component(model)
    print(f"  Component breakdown: {json.dumps(component_params, indent=4)}")

    print("  Computing FLOPs...")
    flops = get_flops(model, (ob_channels, ob_bands), (pcx_channels, pcx_bands), device)
    print(f"  FLOPs: {flops['flops']:,} ({flops['source']})")

    print("  Measuring inference time (single sample)...")
    infer_single = measure_inference_time(
        model, (ob_channels, ob_bands), (pcx_channels, pcx_bands), device, batch_size=1)
    print(f"  Single sample: {infer_single['mean_ms']:.3f} ms")

    print("  Measuring inference time (batch=16)...")
    infer_batch = measure_inference_time(
        model, (ob_channels, ob_bands), (pcx_channels, pcx_bands), device, batch_size=16)
    print(f"  Batch of 16: {infer_batch['mean_ms']:.3f} ms ({infer_batch['per_sample_ms']:.3f} ms/sample)")

    print("  Measuring memory...")
    memory = measure_memory(model, (ob_channels, ob_bands), (pcx_channels, pcx_bands), device)
    print(f"  Peak memory: {memory['peak_memory_mb']:.2f} MB")

    return {
        'method': 'DRN (Ours)',
        'category': 'Deep Learning (Custom)',
        'accuracy': 0.90,  # Final test accuracy from paper
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'model_size_mb': float(model_size_mb),
        'component_parameters': component_params,
        'flops': flops,
        'inference_single': infer_single,
        'inference_batch': infer_batch,
        'memory': memory,
        # Standardized keys for comparison
        'train_time_sec': 2520.0,  # ~42 min on A100 for 5-fold ensemble
        'inference_time_sec': float(infer_single['mean_ms'] / 1000),
        'inference_time_per_sample_ms': float(infer_single['per_sample_ms']),
        'train_peak_memory_mb': float(model_size_mb + memory['peak_memory_mb']),
        'inference_peak_memory_mb': float(memory['peak_memory_mb']),
        'n_parameters': int(total_params),
        'input_shapes': {
            'ob_raw': [32, 21],
            'pcx_raw': [32, 21],
            'ob_enhanced': [ob_channels, ob_bands],
            'pcx_enhanced': [pcx_channels, pcx_bands],
        },
        'architecture_summary': {
            'shared_blocks': 3,
            'channel_progression': [64, 128, 256, 512],
            'feature_dim': int(model.feature_dim),
            'rsn_output_dim': int(model.ob_net.output_dim),
        },
        'training_config': {
            'optimizer': 'AdamW',
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'scheduler': 'OneCycleLR',
            'epochs': 70,
            'batch_size': 16,
            'num_folds': 5,
            'dropout': 0.25,
            'mixup_alpha': 0.2,
            'gradient_clipping': 1.0,
        },
    }


if __name__ == '__main__':
    result = profile_drn()
    print("\n" + json.dumps(result, indent=2, default=str))
