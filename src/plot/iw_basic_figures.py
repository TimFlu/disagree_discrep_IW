"""Recreate basic paper-style comparisons from saved IW and DIS2 estimates."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.plot.compare_iw_hybrids import load_results, plt


STYLES = {
    'dis2_historical': ('DIS2 historical', 'o', '#0072B2'),
    'dis2_reference': ('DIS2 reference', 's', '#D55E00'),
    'iw_overlap_critic': ('IW overlap (plug-in)', '^', '#009E73'),
    'iw_residual_dis2': ('IW residual DIS2 (plug-in)', 'D', '#CC79A7'),
}


def series(data):
    for method, (label, marker, color) in STYLES.items():
        subset = data[data.prediction_method == method]
        for threshold, group in subset.groupby('iw_threshold', dropna=False, sort=True):
            name = label + (f' W={threshold:g}' if pd.notna(threshold) else '')
            valid = group[(group.status == 'ok') & np.isfinite(group.lower_bound)
                          & np.isfinite(group.trg_accuracy)]
            yield method, threshold, name, marker, color, group, valid


def summarize(data):
    rows = []
    for method, threshold, _, _, _, group, valid in series(data):
        gap = valid.trg_accuracy - valid.lower_bound
        rows.append(dict(prediction_method=method, iw_threshold=threshold,
                         n_total=len(group), n_evaluated=len(valid),
                         n_unsupported=len(group) - len(valid),
                         mae=gap.abs().mean(), empirical_coverage=(gap >= 0).mean()
                         if len(valid) else np.nan))
    return pd.DataFrame(rows)


def save(fig, output, stem):
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def accuracy_axes(ax, data):
    # Include saved out-of-range DIS2 values instead of hiding them at zero.
    values = data.lower_bound.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    ax.set(xlim=(0, 1), ylim=(min(0, values.min() - .02) if len(values) else 0,
                            max(1, values.max() + .02) if len(values) else 1),
           xlabel='Target accuracy', ylabel='Saved accuracy bound / estimate')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, zorder=0)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)


def draw_series(ax, data):
    thresholds = sorted(data.iw_threshold.dropna().unique())
    for _, threshold, label, marker, color, _, valid in series(data):
        index = thresholds.index(threshold) if pd.notna(threshold) else 0
        ax.scatter(valid.trg_accuracy, valid.lower_bound, label=label,
                   marker=marker, s=32 + index * 10, alpha=.65,
                   facecolors=color if index == 0 else 'none', edgecolors=color,
                   linewidths=.7 + index * .3)


def make_figures(data, output, suffix):
    table = summarize(data)
    table.to_csv(output / f'summary{suffix}.csv', index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    draw_series(ax, data)
    accuracy_axes(ax, data)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    save(fig, output, f'compare_methods_acc_vs_pred{suffix}')

    methods = [method for method in STYLES if (data.prediction_method == method).any()]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.1 * len(methods), 4.2),
                             squeeze=False)
    for ax, method in zip(axes.flat, methods):
        subset = data[data.prediction_method == method]
        draw_series(ax, subset)
        accuracy_axes(ax, data)
        ax.set_title(STYLES[method][0])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [label.split(' W=')[-1] if ' W=' in label else 'Saved bound'
                            for label in labels], title='W' if method.startswith('iw_') else None,
                  fontsize=8)
    save(fig, output, f'method_variants{suffix}')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for method, (label, marker, color) in STYLES.items():
        group = table[table.prediction_method == method].sort_values('iw_threshold')
        if group.empty:
            continue
        for ax, metric in zip(axes, ['mae', 'empirical_coverage', 'n_unsupported']):
            if method.startswith('iw_'):
                ax.plot(group.iw_threshold, group[metric], marker=marker, color=color, label=label)
            else:
                ax.axhline(group[metric].iloc[0], color=color, linestyle='--', label=label)
    for ax, ylabel in zip(axes, ['Mean absolute error', 'Empirical coverage', 'Unsupported / invalid rows']):
        ax.set(xlabel='IW threshold W', ylabel=ylabel)
        ax.set_xticks(sorted(data.iw_threshold.dropna().unique()))
        ax.grid(alpha=.2)
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    save(fig, output, f'iw_vary_threshold{suffix}')
    print(f'{suffix}: {len(data)} rows, {int(table.n_evaluated.sum())} evaluated; 3 figures')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', required=True)
    parser.add_argument('--plot_dir', required=True)
    parser.add_argument('--bound_strategy', default='logits')
    parser.add_argument('--methods', nargs='+', choices=list(STYLES), default=list(STYLES))
    parser.add_argument('--iw_thresholds', nargs='+', type=float)
    args = parser.parse_args(argv)
    data = load_results(args.results)
    data = data[(data.bound_strategy == args.bound_strategy)
                & data.prediction_method.isin(args.methods)]
    if args.iw_thresholds:
        data = data[data.iw_threshold.isna() | data.iw_threshold.isin(args.iw_thresholds)]
    if data.empty:
        parser.error('No saved rows match the requested methods and representation.')
    output = Path(args.plot_dir)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'serif', 'text.usetex': False}):
        for da in (False, True):
            subset = data[data.train_method.isin(['DANN', 'CDANN']) == da]
            if not subset.empty:
                make_figures(subset, output, '_' + args.bound_strategy + ('_DA' if da else ''))


if __name__ == '__main__':
    main()
