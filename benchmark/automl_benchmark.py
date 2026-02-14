"""
AutoML benchmarks: Auto-sklearn, H2O, TPOT, AutoGluon.
Each wrapped in try/except for graceful degradation.
Google AutoML handled as theoretical-only.
"""

import time
import json
import tracemalloc
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from benchmark.data_utils import prepare_all_data

# Time budgets (seconds)
DEFAULT_TIME_BUDGET = 600  # 10 minutes


def _run_autosklearn(X_train, y_train, X_test, y_test, time_budget):
    """Run auto-sklearn benchmark."""
    import autosklearn.classification

    tmp_dir = tempfile.mkdtemp(prefix='askl_tmp_')
    out_dir = tempfile.mkdtemp(prefix='askl_out_')
    try:
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_budget,
            per_run_time_limit=max(30, time_budget // 10),
            memory_limit=8192,
            n_jobs=-1,
            seed=42,
            ensemble_size=20,
            tmp_folder=tmp_dir,
            output_folder=out_dir,
        )

        tracemalloc.start()
        t0 = time.perf_counter()
        automl.fit(X_train, y_train)
        train_time = time.perf_counter() - t0
        _, train_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            y_pred = automl.predict(X_test)
            times.append(time.perf_counter() - t0)
        _, infer_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        infer_time = np.median(times)

        return {
            'method': 'Auto-sklearn',
            'category': 'AutoML',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'train_time_sec': float(train_time),
            'inference_time_sec': float(infer_time),
            'inference_time_per_sample_ms': float(infer_time / len(y_test) * 1000),
            'train_peak_memory_mb': float(train_mem / 1e6),
            'inference_peak_memory_mb': float(infer_mem / 1e6),
            'n_parameters': 'N/A (ensemble)',
            'best_model': str(automl.show_models())[:500] if hasattr(automl, 'show_models') else 'N/A',
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


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

        return {
            'method': 'H2O AutoML',
            'category': 'AutoML',
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'train_time_sec': float(train_time),
            'inference_time_sec': float(infer_time),
            'inference_time_per_sample_ms': float(infer_time / len(y_test) * 1000),
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


def _run_tpot(X_train, y_train, X_test, y_test, time_budget):
    """Run TPOT benchmark."""
    from tpot import TPOTClassifier

    tpot = TPOTClassifier(
        max_time_mins=max(1, time_budget / 60),
        cv=5,
        n_jobs=1,
        scorers=['accuracy'],
        early_stop=5,
    )

    tracemalloc.start()
    t0 = time.perf_counter()
    tpot.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    _, train_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    times_list = []
    for _ in range(5):
        t0 = time.perf_counter()
        y_pred = tpot.predict(X_test)
        times_list.append(time.perf_counter() - t0)
    _, infer_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    infer_time = np.median(times_list)

    best_pipeline = ''
    try:
        best_pipeline = str(tpot.fitted_pipeline_)[:500]
    except AttributeError:
        pass

    return {
        'method': 'TPOT',
        'category': 'AutoML',
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'train_time_sec': float(train_time),
        'inference_time_sec': float(infer_time),
        'inference_time_per_sample_ms': float(infer_time / len(y_test) * 1000),
        'train_peak_memory_mb': float(train_mem / 1e6),
        'inference_peak_memory_mb': float(infer_mem / 1e6),
        'n_parameters': 'N/A (pipeline)',
        'best_pipeline': best_pipeline,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
    }


def _run_autogluon(X_train, y_train, X_test, y_test, time_budget):
    """Run AutoGluon benchmark."""
    from autogluon.tabular import TabularPredictor

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
            train_df, time_limit=time_budget, presets='best_quality'
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

        return {
            'method': 'AutoGluon',
            'category': 'AutoML',
            'accuracy': float(accuracy_score(y_test, y_pred_np)),
            'train_time_sec': float(train_time),
            'inference_time_sec': float(infer_time),
            'inference_time_per_sample_ms': float(infer_time / len(y_test) * 1000),
            'train_peak_memory_mb': float(train_mem / 1e6),
            'inference_peak_memory_mb': float(infer_mem / 1e6),
            'n_parameters': 'N/A (ensemble)',
            'leaderboard': leaderboard,
            'classification_report': classification_report(y_test, y_pred_np, output_dict=True),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _google_automl_theoretical():
    """Google AutoML Tables: theoretical analysis only (cloud-only service)."""
    return {
        'method': 'Google AutoML Tables',
        'category': 'AutoML (Cloud)',
        'accuracy': None,
        'note': 'Cloud-only service. Cannot benchmark locally. '
                'Based on published literature, Google AutoML Tables typically achieves '
                'comparable accuracy to AutoGluon/H2O on tabular datasets. '
                'Training time depends on cloud configuration (typically 1-8 hours). '
                'Cost: ~$19.32/hour for training.',
        'theoretical_complexity': {
            'training': 'O(n * d * T_search) — neural architecture search budget',
            'inference': 'O(d * model_size) — varies by selected architecture',
            'memory': 'Cloud-managed, typically 2-16 GB',
        },
        'train_time_sec': None,
        'inference_time_sec': None,
        'inference_time_per_sample_ms': None,
        'train_peak_memory_mb': None,
        'inference_peak_memory_mb': None,
        'n_parameters': 'N/A (cloud-managed)',
    }


AUTOML_RUNNERS = {
    'Auto-sklearn': _run_autosklearn,
    'H2O AutoML': _run_h2o,
    'TPOT': _run_tpot,
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
                'train_peak_memory_mb': None, 'inference_peak_memory_mb': None,
                'n_parameters': None,
            })

    # Always include Google AutoML theoretical entry
    results.append(_google_automl_theoretical())
    return results


if __name__ == '__main__':
    results = run_all_automl()
    print("\n" + json.dumps(results, indent=2, default=str))
