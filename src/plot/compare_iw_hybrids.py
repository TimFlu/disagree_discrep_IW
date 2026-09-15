"""Plot saved estimates directly; never reconstruct a DIS2 formula."""
import argparse
import inspect
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class _CompatStringArray(pd.arrays.StringArray):
    def __setstate__(self, state):
        # Recent pandas omits the empty attribute dict in this pickle state.
        if len(state) == 2:
            state = (*state, {})
        super().__setstate__(state)


def _compat_string_dtype(storage='python', na_value=pd.NA):
    # Older pandas uses pd.NA for all string dtypes.
    return pd.StringDtype(storage=storage)


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Read newer NumPy/pandas result arrays with the older plotting stack."""

    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = 'numpy.core' + module[len('numpy._core'):]
        if 'na_value' not in inspect.signature(pd.StringDtype).parameters:
            if module in ('pandas', 'pandas.core.arrays.string_') and name == 'StringDtype':
                return _compat_string_dtype
            if module in ('pandas.arrays', 'pandas.core.arrays.string_') and name == 'StringArray':
                return _CompatStringArray
        return super().find_class(module, name)


def _read_results_pickle(path):
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as exc:
        if exc.name != 'numpy._core' and not (exc.name or '').startswith('numpy._core.'):
            raise
        with open(path, 'rb') as handle:
            return _NumpyCompatUnpickler(handle).load()


def load_results(paths):
    frames = []
    for path in paths:
        frame = _read_results_pickle(path).copy()
        if 'prediction_method' not in frame:
            frame['prediction_method'] = 'dis2_historical'
        if 'iw_threshold' not in frame:
            frame['iw_threshold'] = np.nan
        if 'status' not in frame:
            frame['status'] = 'ok'
        if 'estimate_kind' not in frame:
            frame['estimate_kind'] = 'dis2_with_correction'
        frame['input_file'] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', nargs='+', required=True)
    p.add_argument('--plot_dir', required=True)
    p.add_argument('--bound_strategy', default='logits')
    p.add_argument('--DA', action='store_true')
    args = p.parse_args(argv)
    all_data = load_results(args.results)
    data = all_data[all_data['bound_strategy'] == args.bound_strategy].copy()
    da = data['train_method'].isin(['DANN', 'CDANN'])
    data = data[da if args.DA else ~da]
    if data.empty:
        raise SystemExit('No rows match representation/domain-adaptation filters.')
    output = Path(args.plot_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    summary = []
    keys = ['prediction_method', 'iw_threshold', 'estimate_kind']
    for (method, threshold, kind), group in data.groupby(keys, dropna=False):
        valid = group[(group['status'] == 'ok') & np.isfinite(group['lower_bound'])]
        label = method + (f' W={threshold:g}' if pd.notna(threshold) else '')
        if kind == 'plugin':
            label += ' (plug-in)'
        ax.scatter(valid['trg_accuracy'], valid['lower_bound'], label=label, alpha=.65, s=22)
        diff = valid['trg_accuracy'] - valid['lower_bound']
        summary.append({'prediction_method': method, 'iw_threshold': threshold, 'estimate_kind': kind,
                        'n_total': len(group), 'n_evaluated': len(valid), 'n_unsupported': len(group) - len(valid),
                        'empirical_coverage': float((diff >= 0).mean()), 'mae': float(diff.abs().mean()),
                        'mean_signed_gap': float(diff.mean()), 'mean_excess_error': float(diff.clip(lower=0).mean()),
                        'mean_violation_magnitude': float((-diff).clip(lower=0).mean())})
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Actual target accuracy',
           ylabel='Saved accuracy lower bound / plug-in estimate')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    stem = f'compare_iw_hybrids_{args.bound_strategy}' + ('_DA' if args.DA else '')
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{stem}.{suffix}', bbox_inches='tight')
    plt.close(fig)
    table = pd.DataFrame(summary)
    table.to_csv(output / f'{stem}_summary.csv', index=False)
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
