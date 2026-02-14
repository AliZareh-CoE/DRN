"""
Theoretical complexity analysis for all methods.
Computes Big-O complexity and scaling projections.
"""

import math
import numpy as np


def theoretical_complexity(n=1140, d=1344, K=4):
    """
    Compute theoretical Big-O for all methods.

    Args:
        n: number of training samples (default 1140)
        d: number of features (default 1344 for flattened OB+PCx)
        K: number of classes (default 4)
    """
    methods = {}

    # --- Classical ML ---

    methods['Logistic Regression'] = {
        'training_complexity': r'$O(n \cdot d \cdot K \cdot I)$',
        'training_bigo': 'O(n*d*K*I)',
        'inference_complexity': r'$O(d \cdot K)$',
        'inference_bigo': 'O(d*K)',
        'memory_complexity': r'$O(d \cdot K)$',
        'memory_bigo': 'O(d*K)',
        'training_ops': n * d * K * 100,  # ~100 iterations typical
        'inference_ops': d * K,
        'memory_elements': d * K,
        'scales_with_n_inference': False,
    }

    methods['Decision Tree'] = {
        'training_complexity': r'$O(n \cdot d \cdot \log n)$',
        'training_bigo': 'O(n*d*log(n))',
        'inference_complexity': r'$O(\log n)$',
        'inference_bigo': 'O(log(n))',
        'memory_complexity': r'$O(\text{nodes})$',
        'memory_bigo': 'O(nodes)',
        'training_ops': n * d * int(math.log2(max(n, 2))),
        'inference_ops': int(math.log2(max(n, 2))),
        'memory_elements': min(n, 2**20) * 5,
        'scales_with_n_inference': False,
    }

    T_trees, d_sub = 100, int(math.sqrt(d))
    methods['Random Forest'] = {
        'training_complexity': r'$O(T \cdot n \cdot \sqrt{d} \cdot \log n)$',
        'training_bigo': 'O(T*n*sqrt(d)*log(n))',
        'inference_complexity': r'$O(T \cdot \log n)$',
        'inference_bigo': 'O(T*log(n))',
        'memory_complexity': r'$O(T \cdot \text{nodes})$',
        'memory_bigo': 'O(T*nodes)',
        'training_ops': T_trees * n * d_sub * int(math.log2(max(n, 2))),
        'inference_ops': T_trees * int(math.log2(max(n, 2))),
        'memory_elements': T_trees * min(n, 2**20) * 5,
        'scales_with_n_inference': False,
    }

    n_sv_est = int(n * 0.3)
    methods['SVM (RBF)'] = {
        'training_complexity': r'$O(n^2 \cdot d)$',
        'training_bigo': 'O(n^2*d)',
        'inference_complexity': r'$O(n_{sv} \cdot d)$',
        'inference_bigo': 'O(n_sv*d)',
        'memory_complexity': r'$O(n^2)$',
        'memory_bigo': 'O(n^2)',
        'training_ops': n * n * d,
        'inference_ops': n_sv_est * d,
        'memory_elements': n * n,
        'scales_with_n_inference': True,
        'n_sv_fraction': 0.3,
    }

    mlp_params = d * 512 + 512 * 256 + 256 * 128 + 128 * K
    # plus biases
    mlp_params += 512 + 256 + 128 + K
    methods['MLP Classifier'] = {
        'training_complexity': r'$O(n \cdot P \cdot E)$',
        'training_bigo': 'O(n*P*E)',
        'inference_complexity': r'$O(P)$',
        'inference_bigo': 'O(P)',
        'memory_complexity': r'$O(P)$',
        'memory_bigo': 'O(P)',
        'training_ops': n * mlp_params * 100,  # ~100 epochs typical
        'inference_ops': mlp_params,
        'memory_elements': mlp_params,
        'n_parameters_theoretical': mlp_params,
        'scales_with_n_inference': False,
    }

    # --- AutoML ---

    methods['H2O AutoML'] = {
        'training_complexity': r'$O(B \cdot C_{model})$',
        'training_bigo': 'O(B*C_model)',
        'inference_complexity': r'$O(C_{stacked})$',
        'inference_bigo': 'O(C_stacked)',
        'memory_complexity': r'$O(n \cdot d + M_{models})$',
        'memory_bigo': 'O(n*d + M_models)',
        'training_ops': 30 * n * d,  # ~30 model fits
        'inference_ops': 10 * d,  # stacked ensemble
        'memory_elements': n * d + 30 * d,
        'note': 'Trains GBMs, DRFs, DNNs, GLMs, stacked ensembles',
        'scales_with_n_inference': False,
    }

    methods['TPOT'] = {
        'training_complexity': r'$O(G \cdot P_{size} \cdot C_{pipe})$',
        'training_bigo': 'O(G*P*C_pipe)',
        'inference_complexity': r'$O(C_{best})$',
        'inference_bigo': 'O(C_best)',
        'memory_complexity': r'$O(P_{size} \cdot M_{pipe})$',
        'memory_bigo': 'O(P*M_pipe)',
        'training_ops': 100 * 50 * n * d,  # 100 gens * 50 pop * pipeline eval
        'inference_ops': d * 10,  # single best pipeline
        'memory_elements': 50 * d,
        'note': 'Genetic programming over sklearn pipelines',
        'scales_with_n_inference': False,
    }

    methods['AutoGluon'] = {
        'training_complexity': r'$O(B \cdot C_{multi})$',
        'training_bigo': 'O(B*C_multi)',
        'inference_complexity': r'$O(C_{stacked})$',
        'inference_bigo': 'O(C_stacked)',
        'memory_complexity': r'$O(\sum M_{models})$',
        'memory_bigo': 'O(sum(M_models))',
        'training_ops': 20 * n * d,  # ~20 models across layers
        'inference_ops': 15 * d,  # multi-layer stacking
        'memory_elements': 20 * d,
        'note': 'Multi-layer stacking with GBM, NN, kNN base models',
        'scales_with_n_inference': False,
    }

    methods['Google AutoML Tables'] = {
        'training_complexity': r'$O(NAS \cdot C_{model})$',
        'training_bigo': 'O(NAS*C_model)',
        'inference_complexity': r'$O(C_{selected})$',
        'inference_bigo': 'O(C_selected)',
        'memory_complexity': 'Cloud-managed',
        'memory_bigo': 'Cloud',
        'training_ops': None,
        'inference_ops': None,
        'memory_elements': None,
        'note': 'Neural Architecture Search; cloud-only, cost ~$19/hr',
        'scales_with_n_inference': False,
    }

    # --- DRN ---
    drn_params = 7_700_000  # Will be updated with actual measurement
    methods['DRN (Ours)'] = {
        'training_complexity': r'$O(F \cdot E \cdot n \cdot P)$',
        'training_bigo': 'O(F*E*n*P)',
        'inference_complexity': r'$O(P)$',
        'inference_bigo': 'O(P)',
        'memory_complexity': r'$O(P)$',
        'memory_bigo': 'O(P)',
        'training_ops': 5 * 70 * 912 * drn_params,  # 5 folds * 70 epochs * ~912 samples/fold * P
        'inference_ops': drn_params,
        'memory_elements': drn_params,
        'scales_with_n_inference': False,
        'advantages': [
            'Preserves 2D spatial-spectral structure (32x21)',
            'Dual-stream processes OB and PCx independently before fusion',
            'Attention mechanisms learn region-specific importance',
            'Constant inference time regardless of training set size',
        ],
    }

    return methods


def compute_scaling_projections(measured_results):
    """
    Project how each method scales with dataset size n.
    Used for asymptotic growth analysis plots.
    """
    n_values = np.array([100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000])
    d_values = np.array([100, 200, 500, 1000, 1344, 2000, 5000, 10000])
    n_ref, d_ref = 1140, 1344

    projections = {}
    for result in measured_results:
        name = result.get('method', 'Unknown')
        if result.get('train_time_sec') is None:
            continue

        t_train = result['train_time_sec']
        t_infer = result.get('inference_time_sec', 0.001)
        m_mem = result.get('train_peak_memory_mb', 1.0)

        # Scaling exponents depend on method
        if name == 'Logistic Regression':
            train_vs_n = [t_train * (n / n_ref) for n in n_values]
            infer_vs_n = [t_infer for _ in n_values]  # constant
            mem_vs_n = [m_mem for _ in n_values]  # O(d*K), constant w.r.t. n
            train_vs_d = [t_train * (d / d_ref) for d in d_values]
            infer_vs_d = [t_infer * (d / d_ref) for d in d_values]
        elif name == 'Decision Tree':
            train_vs_n = [t_train * (n / n_ref) * math.log2(max(n, 2)) / math.log2(n_ref) for n in n_values]
            infer_vs_n = [t_infer for _ in n_values]  # O(depth), roughly constant
            mem_vs_n = [m_mem * (n / n_ref) for n in n_values]
            train_vs_d = [t_train * (d / d_ref) for d in d_values]
            infer_vs_d = [t_infer for _ in d_values]
        elif name == 'Random Forest':
            train_vs_n = [t_train * (n / n_ref) * math.log2(max(n, 2)) / math.log2(n_ref) for n in n_values]
            infer_vs_n = [t_infer for _ in n_values]  # O(T*depth), roughly constant
            mem_vs_n = [m_mem * (n / n_ref) for n in n_values]
            train_vs_d = [t_train * math.sqrt(d / d_ref) for d in d_values]
            infer_vs_d = [t_infer for _ in d_values]
        elif name == 'SVM (RBF)':
            train_vs_n = [t_train * (n / n_ref) ** 2 for n in n_values]
            infer_vs_n = [t_infer * (n / n_ref) for n in n_values]  # O(n_sv*d), n_sv ~ 0.3n
            mem_vs_n = [m_mem * (n / n_ref) ** 2 for n in n_values]  # O(n^2) kernel matrix
            train_vs_d = [t_train * (d / d_ref) for d in d_values]
            infer_vs_d = [t_infer * (d / d_ref) for d in d_values]
        elif name == 'MLP Classifier':
            train_vs_n = [t_train * (n / n_ref) for n in n_values]
            infer_vs_n = [t_infer for _ in n_values]  # constant
            mem_vs_n = [m_mem for _ in n_values]
            train_vs_d = [t_train * (d / d_ref) for d in d_values]
            infer_vs_d = [t_infer * (d / d_ref) for d in d_values]
        elif 'DRN' in name:
            train_vs_n = [t_train * (n / n_ref) for n in n_values]
            infer_vs_n = [t_infer for _ in n_values]  # CONSTANT - key advantage
            mem_vs_n = [m_mem for _ in n_values]  # CONSTANT - O(P)
            train_vs_d = [t_train for _ in d_values]  # architecture-dependent, not d
            infer_vs_d = [t_infer for _ in d_values]
        else:
            # AutoML: roughly linear to super-linear in n
            train_vs_n = [t_train * (n / n_ref) ** 1.3 for n in n_values]
            infer_vs_n = [t_infer for _ in n_values]
            mem_vs_n = [m_mem * (n / n_ref) for n in n_values]
            train_vs_d = [t_train * (d / d_ref) for d in d_values]
            infer_vs_d = [t_infer * (d / d_ref) for d in d_values]

        projections[name] = {
            'train_vs_n': train_vs_n,
            'infer_vs_n': infer_vs_n,
            'mem_vs_n': mem_vs_n,
            'train_vs_d': train_vs_d,
            'infer_vs_d': infer_vs_d,
            'n_values': n_values.tolist(),
            'd_values': d_values.tolist(),
        }

    return projections


if __name__ == '__main__':
    import json
    theo = theoretical_complexity()
    print(json.dumps(theo, indent=2, default=str))
