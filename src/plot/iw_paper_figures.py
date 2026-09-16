"""Paper figures with an IW hybrid replacing DIS2; use saved predictions verbatim."""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.lib.consts import PCA_FACTORS
from src.plot.compare_iw_hybrids import _read_results_pickle, plt

METHODS = {
    'iw_overlap_critic': 'IW overlap critic',
    'iw_residual_dis2': 'IW residual DIS2',
}
BASELINES = {'ATC_NE': 'ATC (neg. entropy)', 'ATC': 'ATC (max confidence)',
             'COT': 'COT', 'AC': 'AC', 'DOC': 'DoC'}
CASES = ['dataset', 'shift', 'train_method']
REPRESENTATIONS = ['logits', 'features'] + [f'PCA{i}' for i in PCA_FACTORS]


def read_frame(path):
    path = Path(path)
    frame = pd.read_csv(path) if path.suffix.lower() == '.csv' else _read_results_pickle(path)
    return frame.copy()


def require_columns(frame, columns, description):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{description} missing columns: {sorted(missing)}')


def normalize_cases(frame):
    frame = frame.copy()
    for column in CASES:
        if frame[column].isna().any():
            raise ValueError(f'Missing experiment identifier: {column}')
        frame[column] = frame[column].astype(str)
    # A CSV reader can infer integer-valued shift IDs as floats.
    frame['shift'] = frame['shift'].str.replace(r'^(\d+)\.0$', r'\1', regex=True)
    return frame


def ensure_unique(frame, keys, description):
    if frame.duplicated(keys).any():
        raise ValueError(f'Duplicate {description} experiment keys. Select one run per '
                         'method/threshold/seed/representation; do not combine overlapping runs.')


def load_hybrids(paths, methods, thresholds=None, seeds=None, prediction_column='lower_bound'):
    data = pd.concat([read_frame(p) for p in paths], ignore_index=True)
    require_columns(data, CASES + ['prediction_method', 'iw_threshold', 'seed',
                    'bound_strategy', 'status', 'trg_accuracy', prediction_column], 'Hybrid results')
    data = normalize_cases(data)
    data = data[data.prediction_method.isin(methods)].copy()
    if thresholds is not None:
        data = data[data.iw_threshold.isin(thresholds)]
    if seeds is not None:
        data = data[data.seed.isin(seeds)]
    if data.empty:
        raise ValueError('No hybrid rows match the requested methods, thresholds, and seeds.')
    ensure_unique(data, CASES + ['prediction_method', 'iw_threshold', 'seed', 'bound_strategy'], 'hybrid')
    # No DIS2 reconstruction, epsilon subtraction, clipping, or max-over-seed selection.
    data['prediction'] = pd.to_numeric(data[prediction_column], errors='raise')
    data['trg_accuracy'] = pd.to_numeric(data.trg_accuracy, errors='raise')
    return data


def load_baselines(paths, methods, temperature):
    data = pd.concat([read_frame(p) for p in paths], ignore_index=True)
    require_columns(data, CASES + ['prediction_method', 'temperature', 'lower_bound',
                                  'trg_accuracy'], 'Baseline results')
    data = normalize_cases(data)
    data = data[data.prediction_method.isin(methods) & data.temperature.eq(temperature)].copy()
    data['prediction'] = pd.to_numeric(data.lower_bound, errors='raise')
    data['trg_accuracy'] = pd.to_numeric(data.trg_accuracy, errors='raise')
    if 'status' not in data:
        data['status'] = 'ok'
    ensure_unique(data, CASES + ['prediction_method'], 'baseline')
    return data


def case_index(data):
    return pd.MultiIndex.from_frame(data[CASES])


def matched_comparison(hybrid, baselines, methods):
    """Same available dataset/shift/model cases for every comparison series."""
    common = case_index(hybrid)
    selected = []
    for method in methods:
        group = baselines[baselines.prediction_method == method].copy()
        if group.empty:
            raise ValueError(f'No {method} baseline rows for this DA/temperature selection. '
                             'Run src.eval.other_methods or select available --baseline_methods.')
        common = common.intersection(case_index(group), sort=False)
        selected.append(group)
    if common.empty:
        raise ValueError('No common dataset/shift/train_method cases between hybrid and baselines.')
    result = [hybrid[case_index(hybrid).isin(common)].copy()]
    for group in selected:
        group = group[case_index(group).isin(common)].copy()
        check = result[0][CASES + ['trg_accuracy']].merge(
            group[CASES + ['trg_accuracy']], on=CASES, suffixes=('_hybrid', '_baseline'),
            validate='one_to_one')
        if not np.allclose(check.trg_accuracy_hybrid, check.trg_accuracy_baseline,
                           rtol=0, atol=1e-6, equal_nan=False):
            raise ValueError('Target accuracies differ on matched experiment keys. '
                             'Use baseline and hybrid results from the same feature data.')
        result.append(group)
    return result


def plotted_rows(group):
    return (group.status.eq('ok') & np.isfinite(group.prediction)
            & np.isfinite(group.trg_accuracy))


def label_representation(name):
    if name == 'logits':
        return 'Logits'
    if name == 'features':
        return 'Full features (unprojected)'
    if name == 'PCA1':
        return 'PCA1 (all available SVD columns)'
    return f'PCA: top 1/{name[3:]} of input dimension'


def draw_axes(ax, groups):
    values = np.concatenate([g.prediction[np.isfinite(g.prediction)].to_numpy()
                             for g in groups]) if groups else np.array([])
    lo = min(0., float(values.min()) - .02) if len(values) else 0.
    hi = max(1., float(values.max()) + .02) if len(values) else 1.
    ax.set(xlim=(0, 1), ylim=(lo, hi), xlabel='Target accuracy',
           ylabel='Predicted target accuracy')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.grid(alpha=.2)
    ax.set_axisbelow(True)


def scatter(ax, group, label, **kwargs):
    valid = group[plotted_rows(group)]
    ax.scatter(valid.trg_accuracy, valid.prediction, label=label, **kwargs)
    return len(valid)


def save_figure(fig, output, stem):
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def record(group, plot_name, representation, label, n_available=None):
    points = group.copy()
    points['plot'] = plot_name
    points['panel'] = representation
    points['series'] = label
    points['plotted'] = plotted_rows(points)
    valid = points[points.plotted]
    gap = valid.trg_accuracy - valid.prediction
    summary = dict(plot=plot_name, panel=representation, series=label,
                   n_available=len(group) if n_available is None else n_available,
                   n_matched=len(group), n_plotted=len(valid),
                   n_unsupported_or_nonfinite=len(group) - len(valid),
                   mae=gap.abs().mean(), empirical_coverage=(gap >= 0).mean() if len(gap) else np.nan,
                   mean_violation_magnitude=(-gap).clip(lower=0).mean())
    return points, summary


def accuracy_figure(data, baselines, args, output, stem, title):
    hybrid = data[data.bound_strategy == args.bound_strategy]
    if hybrid.empty:
        raise ValueError(f'No {args.bound_strategy} hybrid results for {title}.')
    groups = matched_comparison(hybrid, baselines, args.baseline_methods)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scatter(ax, groups[0], title + ' (plug-in)', marker='*', s=55, color='#0072B2', zorder=3)
    colors = ['#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    points, summaries = [], []
    for index, group in enumerate(groups):
        method = group.prediction_method.iloc[0]
        label = title + ' (plug-in)' if index == 0 else BASELINES[method]
        if index:
            scatter(ax, group, label, s=19, alpha=.5, color=colors[index-1])
        n_available = len(hybrid) if index == 0 else len(baselines[baselines.prediction_method == method])
        p, summary = record(group, 'accuracy_vs_baselines', args.bound_strategy, label, n_available)
        points.append(p)
        summaries.append(summary)
    draw_axes(ax, groups)
    ax.set_title(label_representation(args.bound_strategy))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    save_figure(fig, output, stem + f'_accuracy_vs_baselines_{args.bound_strategy}')
    return points, summaries


def representation_figure(data, strategies, output, stem, title, plot_name, columns):
    rows = math.ceil(len(strategies) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.1 * rows), squeeze=False)
    groups = [data[data.bound_strategy == strategy] for strategy in strategies]
    points, summaries = [], []
    for ax, strategy, group in zip(axes.flat, strategies, groups):
        draw_axes(ax, groups)
        ax.set_title(label_representation(strategy), fontsize=11)
        p, summary = record(group, plot_name, strategy, title + ' (plug-in)')
        points.append(p)
        summaries.append(summary)
        if group.empty:
            ax.text(.5, .5, 'Not run: ' + strategy, ha='center', va='center', transform=ax.transAxes)
        else:
            count = scatter(ax, group, title + ' (plug-in)', s=22, color='#0072B2', alpha=.7)
            ax.text(.03, .97, f'n={count}/{len(group)}; MAE={summary["mae"]:.3f}\n'
                    f'coverage={summary["empirical_coverage"]:.1%}',
                    va='top', fontsize=9, transform=ax.transAxes)
    for ax in list(axes.flat)[len(strategies):]:
        ax.set_visible(False)
    fig.suptitle(title + ' (plug-in)', fontsize=13)
    save_figure(fig, output, stem + '_' + plot_name)
    return points, summaries


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', nargs='+', required=True, help='Hybrid .pkl or flat .csv files.')
    p.add_argument('--other_results_fname', nargs='+', help='Baseline .pkl/.csv from src.eval.other_methods.')
    p.add_argument('--plot_dir', required=True)
    p.add_argument('--methods', nargs='+', choices=list(METHODS), default=list(METHODS))
    p.add_argument('--iw_thresholds', nargs='+', type=float)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--figures', nargs='+', choices=['accuracy', 'representations'],
                   default=['accuracy', 'representations'])
    p.add_argument('--bound_strategy', default='logits', help='Representation for accuracy-vs-baselines only.')
    p.add_argument('--baseline_methods', nargs='+', choices=list(BASELINES), default=['ATC_NE', 'COT', 'AC'])
    p.add_argument('--temperature', choices=['source', 'none'], default='source')
    p.add_argument('--da', choices=['both', 'yes', 'no'], default='both')
    p.add_argument('--prediction_column', choices=['lower_bound', 'accuracy_estimate_raw'], default='lower_bound')
    p.add_argument('--allow_missing', action='store_true', help='Annotate missing representations instead of failing.')
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        data = load_hybrids(args.results, args.methods, args.iw_thresholds, args.seeds, args.prediction_column)
        if 'accuracy' in args.figures and not args.other_results_fname:
            raise ValueError('Accuracy-vs-baselines requires --other_results_fname from src.eval.other_methods. '
                             'For representation figures only use --figures representations.')
        baselines = load_baselines(args.other_results_fname, args.baseline_methods, args.temperature) \
            if 'accuracy' in args.figures else None
        partitions = []
        for (method, threshold, seed), group in data.groupby(['prediction_method', 'iw_threshold', 'seed']):
            for da in ([False, True] if args.da == 'both' else [args.da == 'yes']):
                subset = group[group.train_method.isin(['DANN', 'CDANN']) == da]
                if subset.empty:
                    continue
                if 'representations' in args.figures:
                    missing = set(REPRESENTATIONS) - set(subset.bound_strategy)
                    if missing and not args.allow_missing:
                        raise ValueError(f'Missing representations for {method}, W={threshold:g}, seed={seed}: '
                                         f'{sorted(missing)}. Rerun evaluation with --bound_strategies '
                                         + ' '.join(REPRESENTATIONS) + ' (or inspect with --allow_missing).')
                baseline_group = baselines[baselines.train_method.isin(['DANN', 'CDANN']) == da] \
                    if baselines is not None else None
                if 'accuracy' in args.figures:
                    selected = subset[subset.bound_strategy == args.bound_strategy]
                    if selected.empty:
                        raise ValueError(f'No {args.bound_strategy} rows for {method}, W={threshold:g}, seed={seed}.')
                    matched_comparison(selected, baseline_group, args.baseline_methods)
                partitions.append((method, threshold, seed, da, subset, baseline_group))
        if not partitions:
            raise ValueError('No rows match the requested DA selection.')
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    output = Path(args.plot_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    with plt.rc_context({'font.family': 'serif', 'text.usetex': False}):
        for method, threshold, seed, da, subset, baseline_group in partitions:
            stem = f'{method}_W{threshold:g}_seed{seed:g}' + ('_DA' if da else '_nonDA')
            stem += '_' + args.prediction_column
            title = f'{METHODS[method]} | W={threshold:g} | seed={seed:g}'
            points, summaries = [], []
            if 'accuracy' in args.figures:
                pts, stats = accuracy_figure(subset, baseline_group, args, output, stem, title)
                points.extend(pts)
                summaries.extend(stats)
            if 'representations' in args.figures:
                for strategies, name, columns in [(['logits', 'features'], 'logits_vs_features', 2),
                       ([f'PCA{i}' for i in PCA_FACTORS], 'all_pca', 3)]:
                    pts, stats = representation_figure(subset, strategies, output, stem, title, name, columns)
                    points.extend(pts)
                    summaries.extend(stats)
            pd.concat(points, ignore_index=True).to_csv(output / f'{stem}_points.csv', index=False)
            summary = pd.DataFrame(summaries).assign(hybrid_method=method, iw_threshold=threshold,
                                                    seed=seed, DA=da, prediction_column=args.prediction_column)
            summary.to_csv(output / f'{stem}_summary.csv', index=False)
            all_summaries.append(summary)
            print(f'Saved {stem}: {len(summaries)} series/panels; predictions used without reconstruction.')
    return pd.concat(all_summaries, ignore_index=True)


if __name__ == '__main__':
    main()
