"""
Classical ML baselines: Logistic Regression, Decision Tree, Random Forest, SVM, MLP.
All use the same train/test split as DRN.
"""

import time
import json
import tracemalloc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

from benchmark.data_utils import prepare_all_data


CLASSICAL_MODELS = {
    'Logistic Regression': lambda: LogisticRegression(
        max_iter=5000, multi_class='multinomial', solver='lbfgs',
        random_state=42, n_jobs=-1
    ),
    'Decision Tree': lambda: DecisionTreeClassifier(
        random_state=42, max_depth=20
    ),
    'Random Forest': lambda: RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=20
    ),
    'SVM (RBF)': lambda: SVC(
        kernel='rbf', random_state=42, probability=True
    ),
    'MLP Classifier': lambda: MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), max_iter=1000, random_state=42,
        early_stopping=True, validation_fraction=0.15
    ),
}


def count_model_params(model, name):
    """Estimate number of parameters for a fitted sklearn model."""
    if name == 'Logistic Regression':
        return model.coef_.size + model.intercept_.size
    elif name == 'Decision Tree':
        return model.tree_.node_count * 5
    elif name == 'Random Forest':
        return sum(t.tree_.node_count * 5 for t in model.estimators_)
    elif name == 'SVM (RBF)':
        return model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size
    elif name == 'MLP Classifier':
        total = sum(c.size for c in model.coefs_)
        total += sum(i.size for i in model.intercepts_)
        return total
    return 0


def get_svm_support_vector_count(model, name):
    """Get number of support vectors for SVM."""
    if name == 'SVM (RBF)':
        return model.n_support_.sum()
    return None


def benchmark_classical_model(name, model_fn, X_train, y_train, X_test, y_test):
    """
    Benchmark a single classical ML model.
    Returns dict with accuracy, timing, memory, parameter count.
    """
    model = model_fn()

    # --- Training ---
    tracemalloc.start()
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    _, train_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- Inference (run multiple times for stable timing) ---
    tracemalloc.start()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        times.append(time.perf_counter() - t0)
    _, infer_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    infer_time = np.median(times)
    accuracy = accuracy_score(y_test, y_pred)
    n_params = count_model_params(model, name)
    n_sv = get_svm_support_vector_count(model, name)

    result = {
        'method': name,
        'category': 'Classical ML',
        'accuracy': float(accuracy),
        'train_time_sec': float(train_time),
        'inference_time_sec': float(infer_time),
        'inference_time_per_sample_ms': float(infer_time / len(y_test) * 1000),
        'train_peak_memory_mb': float(train_peak_mem / 1e6),
        'inference_peak_memory_mb': float(infer_peak_mem / 1e6),
        'n_parameters': int(n_params),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
    }
    if n_sv is not None:
        result['n_support_vectors'] = int(n_sv)
    return result


def run_all_classical(data=None):
    """Run all classical ML benchmarks. Returns list of result dicts."""
    if data is None:
        data = prepare_all_data()

    results = []
    for name, model_fn in CLASSICAL_MODELS.items():
        print(f"  Running {name}...")
        try:
            result = benchmark_classical_model(
                name, model_fn,
                data['X_train_flat'], data['y_train'],
                data['X_test_flat'], data['y_test']
            )
            results.append(result)
            print(f"    Accuracy: {result['accuracy']:.4f}, "
                  f"Train: {result['train_time_sec']:.2f}s, "
                  f"Infer: {result['inference_time_sec']:.4f}s, "
                  f"Params: {result['n_parameters']}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({
                'method': name, 'category': 'Classical ML',
                'accuracy': None, 'error': str(e),
            })
    return results


if __name__ == '__main__':
    results = run_all_classical()
    print("\n" + json.dumps(results, indent=2, default=str))
