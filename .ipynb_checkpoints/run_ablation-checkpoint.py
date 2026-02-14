#!/usr/bin/env python3
"""
Architectural Ablation Study - Multi-GPU Parallel Execution

Usage:
    python run_ablation.py --data_dir /path/to/data --num_gpus 8

This script runs ablation experiments across multiple GPUs using multiprocessing.
"""

import os
import sys
import csv
import json
import time
import random
import argparse
import traceback
from datetime import datetime
from collections import defaultdict
import multiprocessing as mp

import numpy as np
import torch

from config import (
    CONFIG, NUM_RUNS_PER_EXPERIMENT, RUN_SEEDS, NUM_GPUS, OUTPUT_DIR,
    ARCHITECTURAL_ABLATION_CONFIGS, OB_PKL_PATH, PCX_PKL_PATH, META_CSV_PATH
)
from data import load_data, create_balanced_odor_groups
from train import train_ensemble


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_experiment(gpu_id, exp_name, model_kwargs, seed, X_ob, X_pcx, y, config):
    """Run a single experiment on a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    set_seed(seed)
    
    result = {
        'experiment': exp_name,
        'seed': seed,
        'gpu_id': gpu_id,
        'model_kwargs': str(model_kwargs),
        'val_mean': None,
        'val_std': None,
        'val_sem': None,
        'val_accs': None,
        'test_acc': None,
        'status': 'running',
        'error': None,
        'start_time': datetime.now().isoformat(),
    }
    
    try:
        print(f"[GPU {gpu_id}] Starting: {exp_name} (seed={seed})")
        start_time = time.time()
        
        X_ob_bal, X_pcx_bal, _, y_encoded, _, _, label_enc = \
            create_balanced_odor_groups(X_ob, X_pcx, y, selected_ester=config['ester_representative'])
        
        _, _, exp_results = train_ensemble(
            X_ob_bal, X_pcx_bal, y_encoded, config, label_enc,
            device=device, model_kwargs=model_kwargs
        )
        
        elapsed = time.time() - start_time
        
        result['val_mean'] = exp_results['val_mean']
        result['val_std'] = exp_results['val_std']
        result['val_sem'] = exp_results['val_sem']
        result['val_accs'] = exp_results['val_accs']
        result['test_acc'] = exp_results['test_acc'] * 100.0
        result['status'] = 'success'
        result['elapsed_minutes'] = elapsed / 60.0
        
        print(f"[GPU {gpu_id}] ✓ {exp_name} (seed={seed}): "
              f"Val={result['val_mean']:.2f}%, Test={result['test_acc']:.2f}% ({elapsed/60:.1f} min)")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"[GPU {gpu_id}] ✗ {exp_name} (seed={seed}) FAILED: {e}")
        traceback.print_exc()
    
    finally:
        torch.cuda.empty_cache()
        result['end_time'] = datetime.now().isoformat()
    
    return result


def worker_fn(gpu_id, task_queue, result_queue, X_ob, X_pcx, y, config):
    """Worker process for a specific GPU."""
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    print(f"[Worker GPU-{gpu_id}] Initialized on {torch.cuda.get_device_name(gpu_id)}")
    
    while True:
        task = task_queue.get()
        if task is None:  # Poison pill
            break
        
        exp_name, model_kwargs, seed = task
        result = run_single_experiment(gpu_id, exp_name, model_kwargs, seed, X_ob, X_pcx, y, config)
        result_queue.put(result)
        torch.cuda.empty_cache()
    
    print(f"[Worker GPU-{gpu_id}] Shutting down")


def aggregate_results(all_results):
    """Aggregate results from multiple runs."""
    by_experiment = defaultdict(list)
    for r in all_results:
        if r['status'] == 'success':
            by_experiment[r['experiment']].append(r)
    
    aggregated = []
    baseline_test = baseline_val = None
    
    for exp_name, _ in ARCHITECTURAL_ABLATION_CONFIGS:
        runs = by_experiment.get(exp_name, [])
        
        if not runs:
            aggregated.append({
                'experiment': exp_name, 'n_runs': 0,
                'val_mean': None, 'val_std': None, 'val_sem': None,
                'test_mean': None, 'test_std': None, 'test_sem': None,
                'delta_val': None, 'delta_test': None,
            })
            continue
        
        val_means = [r['val_mean'] for r in runs]
        test_accs = [r['test_acc'] for r in runs]
        
        val_mean = np.mean(val_means)
        val_std = np.std(val_means, ddof=1) if len(val_means) > 1 else 0.0
        val_sem = val_std / np.sqrt(len(val_means))
        
        test_mean = np.mean(test_accs)
        test_std = np.std(test_accs, ddof=1) if len(test_accs) > 1 else 0.0
        test_sem = test_std / np.sqrt(len(test_accs))
        
        if baseline_test is None:
            baseline_test, baseline_val = test_mean, val_mean
            delta_val = delta_test = 0.0
        else:
            delta_val = val_mean - baseline_val
            delta_test = test_mean - baseline_test
        
        aggregated.append({
            'experiment': exp_name, 'n_runs': len(runs),
            'val_mean': val_mean, 'val_std': val_std, 'val_sem': val_sem,
            'test_mean': test_mean, 'test_std': test_std, 'test_sem': test_sem,
            'delta_val': delta_val, 'delta_test': delta_test,
        })
    
    return aggregated


def save_incremental_results(all_results, output_dir):
    """Save results incrementally (overwrites checkpoint file)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save checkpoint JSON (overwritten each time)
    checkpoint_path = os.path.join(output_dir, 'ablation_checkpoint.json')
    json_results = []
    for r in all_results:
        r_copy = r.copy()
        if r_copy.get('val_accs'):
            r_copy['val_accs'] = [float(v) for v in r_copy['val_accs']]
        json_results.append(r_copy)
    with open(checkpoint_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    # Also save checkpoint CSV
    csv_path = os.path.join(output_dir, 'ablation_checkpoint.csv')
    fieldnames = ['experiment', 'seed', 'gpu_id', 'val_mean', 'val_std', 'val_sem', 
                  'test_acc', 'status', 'error', 'elapsed_minutes', 'model_kwargs']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)


def save_results(all_results, aggregated, output_dir):
    """Save detailed and summary results."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Detailed CSV
    csv_path = os.path.join(output_dir, f'ablation_detailed_{timestamp}.csv')
    fieldnames = ['experiment', 'seed', 'gpu_id', 'val_mean', 'val_std', 'val_sem', 
                  'test_acc', 'status', 'error', 'elapsed_minutes', 'model_kwargs']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[SAVED] Detailed: {csv_path}")
    
    # JSON
    json_path = os.path.join(output_dir, f'ablation_detailed_{timestamp}.json')
    json_results = []
    for r in all_results:
        r_copy = r.copy()
        if r_copy.get('val_accs'):
            r_copy['val_accs'] = [float(v) for v in r_copy['val_accs']]
        json_results.append(r_copy)
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"[SAVED] JSON: {json_path}")
    
    # Summary CSV
    summary_path = os.path.join(output_dir, f'ablation_summary_{timestamp}.csv')
    fieldnames = ['experiment', 'n_runs', 'val_mean', 'val_std', 'val_sem',
                  'test_mean', 'test_std', 'test_sem', 'delta_val', 'delta_test']
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in aggregated:
            row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items() if k in fieldnames}
            writer.writerow(row)
    print(f"[SAVED] Summary: {summary_path}")
    
    # Latest copy
    latest_path = os.path.join(output_dir, 'ablation_summary_latest.csv')
    with open(latest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in aggregated:
            row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items() if k in fieldnames}
            writer.writerow(row)
    print(f"[SAVED] Latest: {latest_path}")


def print_summary(aggregated):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("ARCHITECTURAL ABLATION RESULTS")
    print("=" * 100)
    print(f"{'Experiment':<32} | {'N':>3} | {'Val Acc (%)':>18} | {'Test Acc (%)':>18} | {'Δ Test':>10}")
    print("-" * 100)
    
    for r in aggregated:
        if r['n_runs'] == 0:
            print(f"{r['experiment']:<32} | {'0':>3} | {'FAILED':>18} | {'FAILED':>18} | {'N/A':>10}")
        else:
            val_str = f"{r['val_mean']:.2f} ± {r['val_std']:.2f}"
            test_str = f"{r['test_mean']:.2f} ± {r['test_std']:.2f}"
            delta_str = f"{r['delta_test']:+.2f}" if r['delta_test'] != 0 else "—"
            print(f"{r['experiment']:<32} | {r['n_runs']:>3} | {val_str:>18} | {test_str:>18} | {delta_str:>10}")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Run Architectural Ablation Study')
    parser.add_argument('--num_gpus', type=int, default=NUM_GPUS, help='Number of GPUs to use')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--ob_pkl', type=str, default=None, help='OB pickle path')
    parser.add_argument('--pcx_pkl', type=str, default=None, help='PCx pickle path')
    parser.add_argument('--meta_csv', type=str, default=None, help='Metadata CSV path')
    args = parser.parse_args()
    
    # Set paths - use config.py defaults if not specified on command line
    ob_path = args.ob_pkl or OB_PKL_PATH
    pcx_path = args.pcx_pkl or PCX_PKL_PATH
    meta_path = args.meta_csv or META_CSV_PATH
    
    print("=" * 80)
    print("ARCHITECTURAL ABLATION STUDY")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    X_ob, X_pcx, y, _ = load_data(ob_path, pcx_path, meta_path)
    print(f"Data loaded: OB={X_ob.shape}, PCx={X_pcx.shape}")
    
    # Check GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"\nDetected {torch.cuda.device_count()} GPUs, using {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create tasks
    all_tasks = []
    for exp_name, model_kwargs in ARCHITECTURAL_ABLATION_CONFIGS:
        for seed in RUN_SEEDS[:NUM_RUNS_PER_EXPERIMENT]:
            all_tasks.append((exp_name, model_kwargs, seed))
    
    print(f"\nTotal tasks: {len(all_tasks)} ({len(ARCHITECTURAL_ABLATION_CONFIGS)} experiments × {NUM_RUNS_PER_EXPERIMENT} seeds)")
    print(f"Config: {CONFIG}")
    
    # Run with multiprocessing
    mp.set_start_method('spawn', force=True)
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Start workers
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_fn, args=(gpu_id, task_queue, result_queue, X_ob, X_pcx, y, CONFIG))
        p.start()
        workers.append(p)
    
    # Add tasks
    for task in all_tasks:
        task_queue.put(task)
    
    # Add poison pills
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # Collect results with incremental saving
    all_results = []
    for i in range(len(all_tasks)):
        result = result_queue.get()
        all_results.append(result)
        completed = i + 1
        pct = 100.0 * completed / len(all_tasks)
        print(f"[Progress] {completed}/{len(all_tasks)} ({pct:.1f}%)")
        
        # Incremental save every 5 experiments or on last one
        if completed % 5 == 0 or completed == len(all_tasks):
            save_incremental_results(all_results, args.output_dir)
    
    # Wait for workers
    for p in workers:
        p.join()
    
    print("\nAll experiments completed!")
    
    # Aggregate and save
    aggregated = aggregate_results(all_results)
    save_results(all_results, aggregated, args.output_dir)
    print_summary(aggregated)
    
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)


if __name__ == '__main__':
    main()