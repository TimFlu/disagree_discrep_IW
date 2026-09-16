"""Saved-output comparisons and paper figures for source restriction."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.plot.iw_paper_figures import (read_frame, normalize_cases, ensure_unique, require_columns,
    load_baselines, matched_comparison, accuracy_figure, representation_figure,
    draw_axes, scatter, save_figure, record, plt, CASES, REPRESENTATIONS, BASELINES)

NAMES = {'source_pruning':'Source-pruned DIS2', 'source_weighted_critic':'Source-weighted critic DIS2',
         'source_restriction_dis2_reference':'Full-source DIS2 reference'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', nargs='+', required=True)
    p.add_argument('--plot_dir', required=True)
    p.add_argument('--methods', nargs='+', choices=list(NAMES), default=list(NAMES))
    p.add_argument('--source_thresholds', nargs='+', type=float)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--figures', nargs='+', choices=['comparison','accuracy','representations'], default=['comparison'])
    p.add_argument('--bound_strategy', default='logits')
    p.add_argument('--other_results_fname', nargs='+')
    p.add_argument('--baseline_methods', nargs='+', choices=list(BASELINES), default=['ATC_NE','COT','AC'])
    p.add_argument('--temperature', choices=['source','none'], default='source')
    p.add_argument('--da', choices=['both','yes','no'], default='both')
    p.add_argument('--prediction_column', choices=['lower_bound','accuracy_estimate_raw'], default='lower_bound')
    p.add_argument('--allow_missing', action='store_true')
    args = p.parse_args(argv)
    try:
        data = pd.concat([read_frame(path) for path in args.results],ignore_index=True)
        require_columns(data, CASES+['prediction_method','source_threshold','seed','bound_strategy',
                                    'status','trg_accuracy',args.prediction_column], 'Source restriction')
        data = normalize_cases(data)
        data = data[data.prediction_method.isin(args.methods)].copy()
        if args.source_thresholds is not None:
            data = data[data.source_threshold.isna() | data.source_threshold.isin(args.source_thresholds)]
        if args.seeds is not None:
            data = data[data.seed.isin(args.seeds)]
        if data.empty:
            raise ValueError('No source-restriction results match these filters.')
        ensure_unique(data,CASES+['prediction_method','source_threshold','seed','bound_strategy'],'source restriction')
        data['prediction'] = pd.to_numeric(data[args.prediction_column],errors='raise')
        if 'accuracy' in args.figures and not args.other_results_fname:
            raise ValueError('--figures accuracy requires --other_results_fname from src.eval.other_methods.')
        baseline = load_baselines(args.other_results_fname,args.baseline_methods,args.temperature) \
                   if 'accuracy' in args.figures else None
        partitions=[]
        for seed,group in data.groupby('seed'):
            for da in ([False,True] if args.da=='both' else [args.da=='yes']):
                subset=group[group.train_method.isin(['DANN','CDANN'])==da]
                if subset.empty:
                    continue
                b = baseline[baseline.train_method.isin(['DANN','CDANN'])==da] if baseline is not None else None
                if 'comparison' in args.figures and subset[subset.bound_strategy==args.bound_strategy].empty:
                    raise ValueError(f'No {args.bound_strategy} rows for seed={seed}, DA={da}.')
                for (method,tau),part in subset.groupby(['prediction_method','source_threshold'],dropna=False):
                    if 'representations' in args.figures:
                        missing=set(REPRESENTATIONS)-set(part.bound_strategy)
                        if missing and not args.allow_missing:
                            raise ValueError(f'{method}: missing {sorted(missing)}. Rerun with --bound_strategies '
                                             +' '.join(REPRESENTATIONS)+' or use --allow_missing.')
                    if 'accuracy' in args.figures:
                        matched_comparison(part[part.bound_strategy==args.bound_strategy],b,args.baseline_methods)
                partitions.append((seed,da,subset,b))
        if not partitions:
            raise ValueError('No rows match the DA filter.')
    except (ValueError,FileNotFoundError) as exc:
        p.error(str(exc))
    output=Path(args.plot_dir)
    output.mkdir(parents=True,exist_ok=True)
    suffix=' (conditional DIS2)' if args.prediction_column=='lower_bound' else ' (empirical)'
    summaries=[]
    with plt.rc_context({'text.usetex':False,'font.family':'serif'}):
        for seed,da,subset,baseline in partitions:
            stem=f'source_restriction_seed{seed:g}_'+('DA' if da else 'nonDA')+'_'+args.prediction_column
            pts,stats=[],[]
            if 'comparison' in args.figures:
                chosen=subset[subset.bound_strategy==args.bound_strategy]
                fig,ax=plt.subplots(figsize=(9,5))
                groups=[]
                for (method,tau),part in chosen.groupby(['prediction_method','source_threshold'],dropna=False):
                    title=NAMES[method]+(f' tau={tau:g}' if pd.notna(tau) else '')
                    scatter(ax,part,title+suffix,s=22,alpha=.7)
                    points,summary=record(part,'comparison',args.bound_strategy,title+suffix)
                    groups.append(part); pts.append(points); stats.append(summary)
                draw_axes(ax,groups)
                ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8)
                save_figure(fig,output,stem+'_comparison_'+args.bound_strategy)
            for (method,tau),part in subset.groupby(['prediction_method','source_threshold'],dropna=False):
                name=method+(f'_tau{tau:g}' if pd.notna(tau) else '')
                title=NAMES[method]+(f' | tau={tau:g}' if pd.notna(tau) else '')+f' | seed={seed:g}'
                if 'accuracy' in args.figures:
                    points,summ=accuracy_figure(part,baseline,args,output,stem+'_'+name,title,suffix)
                    pts.extend(points); stats.extend(summ)
                if 'representations' in args.figures:
                    for strategies,plot,cols in [(['logits','features'],'logits_vs_features',2),
                                                  (REPRESENTATIONS[2:],'all_pca',3)]:
                        points,summ=representation_figure(part,strategies,output,stem+'_'+name,title,plot,cols,suffix)
                        pts.extend(points); stats.extend(summ)
            pd.concat(pts,ignore_index=True).to_csv(output/(stem+'_points.csv'),index=False)
            table=pd.DataFrame(stats).assign(seed=seed,DA=da,prediction_column=args.prediction_column)
            table.to_csv(output/(stem+'_summary.csv'),index=False)
            summaries.append(table)
            print(f'Saved {stem}: {len(stats)} panels/series.')
    return pd.concat(summaries,ignore_index=True)


if __name__=='__main__':
    main()
