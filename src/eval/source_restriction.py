"""Evaluate hard source pruning and source-weighted critic training with DIS2."""
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
from src.eval.iw_hybrid import representation
from src.lib.source_restriction import METHODS, restriction_splits, evaluate_source_restriction


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feats_dir', required=True)
    p.add_argument('--results_dir', required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--source_thresholds', nargs='+', type=float, default=[.5, .7, .9])
    p.add_argument('--datasets', nargs='+', choices=list(DATASET_SHIFTS), default=list(DATASET_SHIFTS))
    p.add_argument('--shifts', nargs='+', type=int)
    p.add_argument('--train_methods', nargs='+', choices=TRAIN_METHODS, default=TRAIN_METHODS)
    p.add_argument('--bound_strategies', nargs='+', default=['logits'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--split_fractions', nargs=2, type=float, default=[.4,.1], metavar=('FIT','VALIDATION'))
    p.add_argument('--domain_epochs', type=int, default=100)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--critic_repeats', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--loss_type', choices=['disagreement','DBAT','negative_xent'], default='disagreement')
    p.add_argument('--min_source_samples', type=int, default=2)
    p.add_argument('--include_dis2_reference', action='store_true')
    p.add_argument('--delta', type=float, default=.01)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--threads', type=int, default=1)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ('methods','source_thresholds','datasets','train_methods','bound_strategies','seeds'):
        value = getattr(args,name)
        if len(value) != len(set(value)):
            p.error(f'--{name} must not contain duplicates.')
    if args.threads < 1:
        p.error('--threads must be positive.')
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    config = dict(vars(args), schema_version=1, family='source_restriction')
    run_id = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:12]
    destination = Path(args.results_dir)
    destination.mkdir(parents=True,exist_ok=True)
    result = destination / f'source_restriction_{run_id}.pkl'
    result.with_suffix('.json').write_text(json.dumps(config,indent=2)+'\n')
    rows = []
    for dataset, train_method in itertools.product(args.datasets,args.train_methods):
        for shift in DATASET_SHIFTS[dataset]:
            if args.shifts is not None and shift not in args.shifts:
                continue
            try:
                (sf,sy),(tf,ty),(sl,tl) = load_shift_data(args.feats_dir,dataset,shift,train_method,device)
            except FileNotFoundError:
                print(f'Missing features: {dataset}/{train_method}/{shift}; skipping.')
                continue
            sf,tf,sl,tl = [x.float() for x in (sf,tf,sl,tl)]
            sy,ty = sy.long(),ty.long()
            for seed,strategy in itertools.product(args.seeds,args.bound_strategies):
                fit = restriction_splits(len(sf),seed,args.split_fractions)['fit']
                xs,xt = representation(strategy,sf,tf,sl,tl,fit)
                entries,splits,artifacts = evaluate_source_restriction(xs,sl,sy,xt,tl,
                    methods=args.methods,thresholds=args.source_thresholds,
                    include_reference=args.include_dis2_reference,seed=seed,fractions=args.split_fractions,
                    domain_epochs=args.domain_epochs,epochs=args.epochs,repeats=args.critic_repeats,
                    batch_size=args.batch_size,loss_type=args.loss_type,delta=args.delta,
                    min_source_samples=args.min_source_samples)
                stem = f'{dataset}_{train_method}_{shift}_{strategy}_{seed}_{run_id}'
                split_file = 'source_restriction_splits_'+stem+'.npz'
                score_file = 'source_restriction_scores_'+stem+'.npz'
                model_file = 'source_restriction_models_'+stem+'.pth'
                np.savez(destination/split_file,**{f'{domain}_{part}':idx for domain,parts in splits.items()
                                                  for part,idx in parts.items()})
                np.savez(destination/score_file,source_target_probability=artifacts['source_scores'],
                         **artifacts['source_masks'])
                torch.save(dict(domain=artifacts['domain_state'],critics=artifacts['critics'],
                                strategy=strategy,representation_dim=xs.shape[1],seed=seed),destination/model_file)
                # Target task labels enter only after every method prediction is fixed.
                actual = float((tl.argmax(1)==ty).float().mean())
                ti = splits['target']['eval']
                actual_eval = float((tl[ti].argmax(1)==ty[ti]).float().mean())
                actual_source = float((sl.argmax(1)==sy).float().mean())
                for row in entries:
                    row.update(dataset=dataset,shift=str(shift),train_method=train_method,
                        bound_strategy=strategy,dim=xs.shape[1],src_accuracy=actual_source,h_full_acc=actual_source,
                        trg_accuracy=actual,trg_eval_accuracy=actual_eval,run_id=run_id,
                        split_file=split_file,score_file=score_file,model_file=model_file)
                    ok = row['status']=='ok' and np.isfinite(row['lower_bound'])
                    row['bound_valid'] = bool(row['lower_bound']<=actual) if ok else None
                    row['bound_valid_eval'] = bool(row['lower_bound']<=actual_eval) if ok else None
                    rows.append(row)
                    print(f'{dataset}/{shift}/{train_method}/{strategy}: {row["prediction_method"]} '
                          f'tau={row["source_threshold"]} source={row["n_val_source"]}/{row["n_source_eval_full"]} '
                          f'prediction={row["lower_bound"]:.4f} status={row["status"]}')
                frame = pd.DataFrame(rows)
                frame.to_pickle(result)
                frame.drop(columns=['critic_history'],errors='ignore').to_csv(result.with_suffix('.csv'),index=False)
    if not rows:
        raise SystemExit('No experiments ran. Check the feature path and filters.')
    print(f'Saved {len(rows)} rows to {result}')
    return result


if __name__ == '__main__':
    main()
