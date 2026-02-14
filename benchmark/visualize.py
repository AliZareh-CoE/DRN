"""
Publication-quality visualizations for complexity comparison.
All figures: 300 DPI, PDF + PNG, matplotlib with consistent styling.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# --- Color palette (colorblind-friendly) ---
COLORS = {
    'Logistic Regression': '#4E79A7',
    'Decision Tree':       '#F28E2B',
    'Random Forest':       '#E15759',
    'SVM (RBF)':           '#76B7B2',
    'MLP Classifier':      '#59A14F',
    'H2O AutoML':          '#B07AA1',
    'TPOT':                '#FF9DA7',
    'AutoGluon':           '#9C755F',
    'Google AutoML Tables': '#BAB0AC',
    'DRN (Ours)':          '#D62728',
}

STYLE = {
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def _save_fig(fig, path_stem, output_dir):
    """Save figure as both PNG and PDF."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ['png', 'pdf']:
        path = os.path.join(output_dir, f'{path_stem}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path_stem}.png and .pdf")


def _get_measured(results):
    """Filter to methods with measured values."""
    return [r for r in results if r.get('train_time_sec') is not None]


def plot_absolute_comparison(results, output_dir):
    """Bar charts: Training Time, Inference Time, Memory (absolute scale)."""
    _apply_style()
    measured = _get_measured(results)
    names = [r['method'] for r in measured]
    n = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ('train_time_sec', 'Training Time (s)', 'Training Time'),
        ('inference_time_per_sample_ms', 'Inference Time per Sample (ms)', 'Inference Time'),
        ('train_peak_memory_mb', 'Peak Memory (MB)', 'Peak Memory'),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        values = [r.get(key, 0) or 0 for r in measured]
        colors = [COLORS.get(name, '#999999') for name in names]

        bars = ax.bar(range(n), values, color=colors, edgecolor='black', linewidth=0.5)

        for i, name in enumerate(names):
            if 'DRN' in name:
                bars[i].set_edgecolor('#D62728')
                bars[i].set_linewidth(2.5)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)

        for i, v in enumerate(values):
            if v > 0:
                fmt = f'{v:.1f}' if v >= 1 else f'{v:.3f}'
                ax.text(i, v * 1.02, fmt, ha='center', va='bottom', fontsize=6)

    fig.suptitle('DRN vs. Classical ML & AutoML: Absolute Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_fig(fig, 'complexity_comparison_absolute', output_dir)


def plot_log_comparison(results, output_dir):
    """Same as absolute but with log scale y-axes."""
    _apply_style()
    measured = _get_measured(results)
    names = [r['method'] for r in measured]
    n = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ('train_time_sec', 'Training Time (s)', 'Training Time (log scale)'),
        ('inference_time_per_sample_ms', 'Inference Time per Sample (ms)', 'Inference Time (log scale)'),
        ('train_peak_memory_mb', 'Peak Memory (MB)', 'Peak Memory (log scale)'),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics):
        values = [max(r.get(key, 0) or 0, 1e-6) for r in measured]
        colors = [COLORS.get(name, '#999999') for name in names]

        bars = ax.bar(range(n), values, color=colors, edgecolor='black', linewidth=0.5)
        for i, name in enumerate(names):
            if 'DRN' in name:
                bars[i].set_edgecolor('#D62728')
                bars[i].set_linewidth(2.5)

        ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)

    fig.suptitle('DRN vs. Classical ML & AutoML: Log-Scale Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save_fig(fig, 'complexity_comparison_log', output_dir)


def plot_asymptotic_analysis(projections, measured_results, output_dir):
    """
    2x3 grid: training vs n, inference vs n, memory vs n,
              training vs d, speedup factor, accuracy-efficiency.
    """
    _apply_style()
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    n_ref = 1140

    # --- Panel 1: Training time vs n ---
    ax1 = fig.add_subplot(gs[0, 0])
    for name, proj in projections.items():
        color = COLORS.get(name, '#999999')
        lw = 2.5 if 'DRN' in name else 1.2
        ax1.plot(proj['n_values'], proj['train_vs_n'], '-o', color=color,
                 label=name, markersize=3, linewidth=lw)
    ax1.set_xlabel('Number of Training Samples (n)')
    ax1.set_ylabel('Training Time (s)')
    ax1.set_title('Training Time vs. Dataset Size', fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.axvline(x=n_ref, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.text(n_ref * 1.1, ax1.get_ylim()[0] * 2, f'n={n_ref}', fontsize=7, color='gray')
    ax1.legend(fontsize=6, loc='upper left', ncol=1)

    # --- Panel 2: Inference time vs n ---
    ax2 = fig.add_subplot(gs[0, 1])
    for name, proj in projections.items():
        color = COLORS.get(name, '#999999')
        lw = 2.5 if 'DRN' in name else 1.2
        ax2.plot(proj['n_values'], proj['infer_vs_n'], '-s', color=color,
                 label=name, markersize=3, linewidth=lw)
    ax2.set_xlabel('Number of Training Samples (n)')
    ax2.set_ylabel('Inference Time (s)')
    ax2.set_title('Inference Time vs. Dataset Size', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.axvline(x=n_ref, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

    # Annotate constant-time methods
    for name, proj in projections.items():
        vals = proj['infer_vs_n']
        if abs(vals[-1] - vals[0]) / max(vals[0], 1e-10) < 0.01:
            ax2.annotate(f'{name}: constant', xy=(proj["n_values"][-1], vals[-1]),
                         fontsize=5, color=COLORS.get(name, '#999'), ha='right')
    ax2.legend(fontsize=6, loc='upper left')

    # --- Panel 3: Memory vs n ---
    ax3 = fig.add_subplot(gs[0, 2])
    for name, proj in projections.items():
        color = COLORS.get(name, '#999999')
        lw = 2.5 if 'DRN' in name else 1.2
        ax3.plot(proj['n_values'], proj['mem_vs_n'], '-^', color=color,
                 label=name, markersize=3, linewidth=lw)
    ax3.set_xlabel('Number of Training Samples (n)')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('Memory vs. Dataset Size', fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.axvline(x=n_ref, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax3.legend(fontsize=6, loc='upper left')

    # --- Panel 4: Training time vs d ---
    ax4 = fig.add_subplot(gs[1, 0])
    for name, proj in projections.items():
        color = COLORS.get(name, '#999999')
        lw = 2.5 if 'DRN' in name else 1.2
        ax4.plot(proj['d_values'], proj['train_vs_d'], '-o', color=color,
                 label=name, markersize=3, linewidth=lw)
    ax4.set_xlabel('Number of Features (d)')
    ax4.set_ylabel('Training Time (s)')
    ax4.set_title('Training Time vs. Feature Dimension', fontweight='bold')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.axvline(x=1344, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax4.text(1344 * 1.1, ax4.get_ylim()[0] * 2 if ax4.get_ylim()[0] > 0 else 1,
             'd=1344', fontsize=7, color='gray')
    ax4.legend(fontsize=6, loc='upper left')

    # --- Panel 5: Inference speedup factor ---
    ax5 = fig.add_subplot(gs[1, 1])
    drn_proj = projections.get('DRN (Ours)')
    if drn_proj:
        for name, proj in projections.items():
            if name == 'DRN (Ours)':
                continue
            color = COLORS.get(name, '#999999')
            # Speedup = other_method_time / DRN_time (higher = DRN is faster)
            speedup = [p / max(d, 1e-10) for p, d in
                       zip(proj['infer_vs_n'], drn_proj['infer_vs_n'])]
            ax5.plot(proj['n_values'], speedup, '-d', color=color,
                     label=name, markersize=3, linewidth=1.2)
    ax5.set_xlabel('Number of Training Samples (n)')
    ax5.set_ylabel('Inference Time Ratio (Method / DRN)')
    ax5.set_title('Inference Slowdown vs. DRN', fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax5.axvline(x=n_ref, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax5.legend(fontsize=6, loc='upper left')

    # --- Panel 6: Accuracy vs Training Time ---
    ax6 = fig.add_subplot(gs[1, 2])
    measured = _get_measured(measured_results)
    for r in measured:
        name = r['method']
        acc = r.get('accuracy')
        tt = r.get('train_time_sec')
        if acc is not None and tt is not None:
            color = COLORS.get(name, '#999999')
            marker = '*' if 'DRN' in name else 'o'
            ms = 15 if 'DRN' in name else 8
            ax6.scatter(tt, acc * 100, color=color, marker=marker, s=ms**2,
                        label=name, zorder=5 if 'DRN' in name else 3,
                        edgecolors='black', linewidth=0.5)
    ax6.set_xlabel('Training Time (s)')
    ax6.set_ylabel('Test Accuracy (%)')
    ax6.set_title('Accuracy vs. Training Time', fontweight='bold')
    ax6.set_xscale('log')
    ax6.legend(fontsize=6, loc='lower right')

    fig.suptitle('Asymptotic Complexity Analysis: DRN vs. Baselines',
                 fontsize=15, fontweight='bold', y=1.01)
    _save_fig(fig, 'asymptotic_analysis', output_dir)


def generate_latex_table(results, theoretical, output_dir):
    """Generate LaTeX table for paper inclusion."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Comprehensive comparison of DRN against classical ML and AutoML methods '
                 r'on the 4-class odor discrimination task (n=1,140 training samples, d=1,344 features).}')
    lines.append(r'\label{tab:complexity}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{lccccccc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Method} & \textbf{Category} & \textbf{Acc. (\%)} & '
                 r'\textbf{Train (s)} & \textbf{Infer (ms)} & '
                 r'\textbf{Mem. (MB)} & \textbf{Params} & \textbf{Train Complexity} \\')
    lines.append(r'\midrule')

    prev_cat = None
    for r in results:
        name = r.get('method', '?')
        cat = r.get('category', '?')

        # Add midrule between categories
        if prev_cat and cat != prev_cat:
            lines.append(r'\midrule')
        prev_cat = cat

        acc = f"{r['accuracy']*100:.1f}" if r.get('accuracy') else '--'
        tt = f"{r['train_time_sec']:.1f}" if r.get('train_time_sec') else '--'
        it = f"{r.get('inference_time_per_sample_ms', 0):.3f}" if r.get('inference_time_per_sample_ms') else '--'
        mem = f"{r.get('train_peak_memory_mb', 0):.1f}" if r.get('train_peak_memory_mb') else '--'

        params = r.get('n_parameters', 'N/A')
        if isinstance(params, (int, float)) and params > 0:
            if params >= 1e6:
                params = f"{params/1e6:.1f}M"
            elif params >= 1e3:
                params = f"{params/1e3:.0f}K"
            else:
                params = str(int(params))
        else:
            params = str(params)

        theo = theoretical.get(name, {})
        complexity = theo.get('training_complexity', '--')
        if complexity.startswith('$'):
            pass  # already LaTeX math
        elif complexity != '--':
            complexity = f'${complexity}$'

        # Escape underscores for LaTeX
        name_tex = name.replace('_', r'\_')
        cat_tex = cat.replace('_', r'\_')
        params_tex = params.replace('_', r'\_')

        if 'DRN' in name:
            lines.append(f'\\textbf{{{name_tex}}} & \\textbf{{{cat_tex}}} & '
                         f'\\textbf{{{acc}}} & \\textbf{{{tt}}} & \\textbf{{{it}}} & '
                         f'\\textbf{{{mem}}} & \\textbf{{{params_tex}}} & '
                         f'\\textbf{{{complexity}}} \\\\')
        else:
            lines.append(f'{name_tex} & {cat_tex} & {acc} & {tt} & {it} & '
                         f'{mem} & {params_tex} & {complexity} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    tex_path = os.path.join(output_dir, 'complexity_table.tex')
    with open(tex_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {tex_path}")


def plot_accuracy_comparison(results, output_dir):
    """Standalone accuracy comparison bar chart."""
    _apply_style()
    measured = [r for r in results if r.get('accuracy') is not None]
    names = [r['method'] for r in measured]
    accs = [r['accuracy'] * 100 for r in measured]
    colors = [COLORS.get(n, '#999999') for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(names)), accs, color=colors, edgecolor='black', linewidth=0.5)

    for i, name in enumerate(names):
        if 'DRN' in name:
            bars[i].set_edgecolor('#D62728')
            bars[i].set_linewidth(2.5)

    for i, v in enumerate(accs):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy: DRN vs. Baselines (4-class Odor Task)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    _save_fig(fig, 'accuracy_comparison', output_dir)
