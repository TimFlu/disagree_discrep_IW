"""Evaluate IW hybrids without modifying the historical DIS2 entry point."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from src.lib.consts import DATASET_SHIFTS, TRAIN_METHODS
from src.lib.utils import load_shift_data
from src.lib.hybrid_validation import METHODS, split_indices, evaluate_hybrids, evaluate_dis2_reference


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feats_dir', required=True)
    p.add_argument('--results_dir', required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--iw_thresholds', nargs='+', type=float, default=[1., 2., 5.])
    p.add_argument('--datasets', nargs='+', choices=list(DATASET_SHIFTS), default=list(DATASET_SHIFTS))
    p.add_argument('--shifts', nargs='+', type=int)
    p.add_argument('--train_methods', nargs='+', choices=TRAIN_METHODS, default=TRAIN_METHODS)
    p.add_argument('--bound_strategies', nargs='+', default=['logits'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--domain_epochs', type=int, default=100)
    p.add_argument('--critic_repeats', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--source_strength', type=float, default=1.)
    p.add_argument('--loss_type', choices=['disagreement', 'DBAT', 'negative_xent'], default='disagreement')
    p.add_argument('--split_fractions', nargs=3, type=float, default=[.4, .1, .2],
                   metavar=('DOMAIN_TRAIN', 'DOMAIN_CAL', 'CRITIC_SELECT'),
                   help='Fractions of total data; critic training reuses domain train + calibration; remainder evaluates.')
    p.add_argument('--include_dis2_reference', action='store_true')
    p.add_argument('--delta', type=float, default=.01, help='Only used by the optional DIS2 reference.')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=1)
    return p


def representation(strategy, source, target, source_logits, target_logits, train_indices):
    if strategy == 'logits':
        return source_logits, target_logits
    if strategy == 'features':
        return source, target
    if not strategy.startswith('PCA') or not strategy[3:].isdigit() or int(strategy[3:]) < 1:
        raise ValueError('Strategies must be logits, features, or PCA followed by a positive divisor.')
    k = max(1, source.shape[1] // int(strategy[3:]))
    # Match historical uncentered projection, but fit on domain-training data only.
    _, _, vt = torch.linalg.svd(source[train_indices], full_matrices=False)
    v = vt[:k].T
    return source @ v, target @ v


def main(argv=None):
    args = parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    config = vars(args).copy()
    config['schema_version'] = 3
    config['split_protocol'] = 'shared_training'
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    destination = Path(args.results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / f'iw_hybrid_{run_id}.pkl'
    (destination / f'iw_hybrid_{run_id}.json').write_text(json.dumps(config, indent=2) + '\n')
    rows = []
    for dataset, train_method in itertools.product(args.datasets, args.train_methods):
        for shift in DATASET_SHIFTS[dataset]:
            if args.shifts is not None and shift not in args.shifts:
                continue
            try:
                (sf, sy), (tf, ty), (sl, tl) = load_shift_data(args.feats_dir, dataset, shift, train_method, device)
            except FileNotFoundError:
                print(f'Missing features: {dataset}/{train_method}/{shift}; skipping.')
                continue
            sf, tf, sl, tl = [x.float() for x in (sf, tf, sl, tl)]
            sy, ty = sy.long(), ty.long()
            for seed, strategy in itertools.product(args.seeds, args.bound_strategies):
                fit_indices = split_indices(len(sf), seed, args.split_fractions)['domain_train']
                xs, xt = representation(strategy, sf, tf, sl, tl, fit_indices)
                entries, splits = evaluate_hybrids(xs, sl, sy, xt, tl, methods=args.methods,
                    thresholds=args.iw_thresholds, seed=seed, fractions=args.split_fractions,
                    domain_epochs=args.domain_epochs, epochs=args.epochs, repeats=args.critic_repeats,
                    batch_size=args.batch_size, source_strength=args.source_strength, loss_type=args.loss_type)
                if args.include_dis2_reference:
                    entries.append(evaluate_dis2_reference(xs, sl, sy, xt, tl, seed=seed,
                        fractions=args.split_fractions, epochs=args.epochs, repeats=args.critic_repeats,
                        batch_size=args.batch_size, loss_type=args.loss_type, delta=args.delta))
                # Persist exact split indices once per dataset/seed, shared across strategies.
                split_name = f'splits_{dataset}_{train_method}_{shift}_{seed}_{run_id}.npz'
                np.savez(destination / split_name, **{f'{domain}_{name}': idx
                    for domain, partition in splits.items() for name, idx in partition.items()})
                # Target task labels first enter ONLY after all predictions have been produced.
                target_accuracy = float((tl.argmax(1) == ty).float().mean())
                ti = splits['target']['eval']
                eval_accuracy = float((tl[ti].argmax(1) == ty[ti]).float().mean())
                source_accuracy = float((sl.argmax(1) == sy).float().mean())
                for row in entries:
                    row.update(dataset=dataset, shift=str(shift), train_method=train_method,
                               schema_version=config['schema_version'], bound_strategy=strategy, dim=xs.shape[1], src_accuracy=source_accuracy,
                               h_full_acc=source_accuracy, trg_accuracy=target_accuracy,
                               trg_eval_accuracy=eval_accuracy, split_file=split_name, run_id=run_id)
                    valid = row['status'] == 'ok' and np.isfinite(row['lower_bound'])
                    row['bound_valid'] = bool(row['lower_bound'] <= target_accuracy) if valid else None
                    row['bound_valid_eval'] = bool(row['lower_bound'] <= eval_accuracy) if valid else None
                    rows.append(row)
                    print(f"{dataset}/{shift}/{train_method}/{strategy} {row['prediction_method']} "
                          f"W={row['iw_threshold']} accuracy={target_accuracy:.4f} "
                          f"prediction={row['lower_bound']:.4f} status={row['status']}")
                frame = pd.DataFrame(rows)
                frame.to_pickle(result_path)
                frame.drop(columns=['critic_history'], errors='ignore').to_csv(result_path.with_suffix('.csv'), index=False)
    if not rows:
        raise SystemExit('No experiments ran. Check feature paths and dataset/shift filters.')
    print(f'Saved {len(rows)} rows to {result_path}')
    return result_path


if __name__ == '__main__':
    main()
