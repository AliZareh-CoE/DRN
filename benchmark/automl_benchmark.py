"""
AutoML benchmarks: H2O AutoML, AutoGluon.
"""

import time
import json
import tracemalloc
import tempfile
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from benchmark.data_utils import prepare_all_data

# Time budgets (seconds)
DEFAULT_TIME_BUDGET = 600  # 10 minutes


def _run_h2o(X_train, y_train, X_test, y_test, time_budget):
    """Run H2O AutoML benchmark."""
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init(nthreads=-1, max_mem_size="8G")
    try:
        train_df = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])
        train_df['target'] = y_train.astype(str)
        test_df = pd.DataFrame(X_test, columns=[f'f{i}' for i in range(X_test.shape[1])])
        test_df['target'] = y_test.astype(str)

        h2o_train = h2o.H2OFrame(train_df)
        h2o_test = h2o.H2OFrame(test_df)
        h2o_train['target'] = h2o_train['target'].asfactor()
        h2o_test['target'] = h2o_test['target'].asfactor()

        aml = H2OAutoML(
            max_runtime_secs=time_budget,
            seed=42,
            sort_metric='logloss',
        )

        tracemalloc.start()
        t0 = time.perf_counter()
        aml.train(y='target', training_frame=h2o_train)
        train_time = time.perf_counter() - t0
        _, train_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            preds = aml.leader.predict(h2o_test)
            times.append(time.perf_counter() - t0)
        _, infer_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        infer_time = np.median(times)
        y_pred = preds['predict'].as_data_frame().values.flatten().astype(int)

        # Model size: use peak RAM as proxy (can't count ensemble params)
        model_size_mb = float(train_mem / 1e6)

        return {
            'method': 'H2O AutoML',
            'category': 'AutoML',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'train_time_sec': float(train_time),
            'inference_time_sec': float(infer_time),
            'inference_time_per_sample_ms': float(infer_time / len(y_test) * 1000),
            'model_size_mb': model_size_mb,
            'train_peak_memory_mb': float(train_mem / 1e6),
            'inference_peak_memory_mb': float(infer_mem / 1e6),
            'n_parameters': 'N/A (ensemble)',
            'leaderboard': str(aml.leaderboard.head(5).as_data_frame()),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }
    finally:
        try:
            h2o.cluster().shutdown()
        except Exception:
            pass


def _run_autogluon(X_train, y_train, X_test, y_test, time_budget):
    """Run AutoGluon benchmark."""
    import os
    from autogluon.tabular import TabularPredictor

    # Force XGBoost to CPU mode — its GPU build probes CUDA even with num_gpus=0
    old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    tmpdir = tempfile.mkdtemp(prefix='ag_')
    try:
        train_df = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])
        train_df['target'] = y_train
        test_df = pd.DataFrame(X_test, columns=[f'f{i}' for i in range(X_test.shape[1])])
        test_df['target'] = y_test

        tracemalloc.start()
        t0 = time.perf_counter()
        predictor = TabularPredictor(
            label='target', path=tmpdir, eval_metric='accuracy',
            verbosity=1,
        ).fit(
            train_df, time_limit=time_budget, presets='best_quality',
            num_gpus=0,
        )
        train_time = time.perf_counter() - t0
        _, train_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        times = []
        test_features = test_df.drop(columns=['target'])
        for _ in range(5):
            t0 = time.perf_counter()
            y_pred = predictor.predict(test_features)
            times.append(time.perf_counter() - t0)
        _, infer_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        infer_time = np.median(times)
        y_pred_np = y_pred.values.astype(int)

        leaderboard = ''
        try:
            leaderboard = str(predictor.leaderboard()[:5])
        except Exception:
            pass

        # Model size: use peak RAM as proxy (can't count ensemble params)
        model_size_mb = float(train_mem / 1e6)

        return {
            'method': 'AutoGluon',
            'category': 'AutoML',
            'accuracy': float(accuracy_score(y_test, y_pred_np)),
            'train_time_sec': float(train_time),
            'inference_time_sec': float(infer_time),
            'inference_time_per_sample_ms': float(infer_time / len(y_test) * 1000),
            'model_size_mb': model_size_mb,
            'train_peak_memory_mb': float(train_mem / 1e6),
            'inference_peak_memory_mb': float(infer_mem / 1e6),
            'n_parameters': 'N/A (ensemble)',
            'leaderboard': leaderboard,
            'classification_report': classification_report(y_test, y_pred_np, output_dict=True),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        # Restore CUDA_VISIBLE_DEVICES
        if old_cuda is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)



AUTOML_RUNNERS = {
    'H2O AutoML': _run_h2o,
    'AutoGluon': _run_autogluon,
}


def run_all_automl(data=None, time_budget=DEFAULT_TIME_BUDGET):
    """Run all AutoML benchmarks with graceful degradation."""
    if data is None:
        data = prepare_all_data()

    results = []
    for name, runner in AUTOML_RUNNERS.items():
        print(f"  Attempting {name}...")
        try:
            result = runner(
                data['X_train_flat'], data['y_train'],
                data['X_test_flat'], data['y_test'],
                time_budget
            )
            results.append(result)
            print(f"    Accuracy: {result['accuracy']:.4f}, "
                  f"Train: {result['train_time_sec']:.1f}s")
        except ImportError as e:
            print(f"    SKIPPED (not installed): {e}")
            results.append({
                'method': name, 'category': 'AutoML',
                'accuracy': None, 'error': f'INSTALL_FAILED: {e}',
                'train_time_sec': None, 'inference_time_sec': None,
                'inference_time_per_sample_ms': None,
                'model_size_mb': None,
                'train_peak_memory_mb': None, 'inference_peak_memory_mb': None,
                'n_parameters': None,
            })
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'method': name, 'category': 'AutoML',
                'accuracy': None, 'error': str(e),
                'train_time_sec': None, 'inference_time_sec': None,
                'inference_time_per_sample_ms': None,
                'model_size_mb': None,
                'train_peak_memory_mb': None, 'inference_peak_memory_mb': None,
                'n_parameters': None,
            })

    return results


if __name__ == '__main__':
    results = run_all_automl()
    print("\n" + json.dumps(results, indent=2, default=str))
