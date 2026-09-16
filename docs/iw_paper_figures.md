# Paper figures using the IW methods

`python -m src.plot.iw_paper_figures` reproduces these figure types with a hybrid in place of DIS²:

| Figure | Content |
|---|---|
| `*_accuracy_vs_baselines_logits` | Selected hybrid versus ATC (negative entropy), COT, and AC, as in the original comparison |
| `*_logits_vs_features` | Two panels: logits and unprojected full features |
| `*_all_pca` | Six panels: PCA1, PCA4, PCA16, PCA32, PCA64, PCA128 |

Every figure is saved as PNG and PDF. Both hybrids are supported, with separate files for each method, threshold, seed, and DA/non-DA group. DIS² is not plotted in these figures. The historical plotting scripts remain available for reproducing historical results.

## 1. Generate the representations

From the repository root in your existing training environment:

```bash
python -m src.eval.iw_hybrid \
  --feats_dir data/features \
  --results_dir results/iw_all_representations \
  --methods iw_overlap_critic iw_residual_dis2 \
  --bound_strategies logits features PCA1 PCA4 PCA16 PCA32 PCA64 PCA128 \
  --iw_thresholds 1 2 5 --seeds 0 \
  --include_dis2_reference
```

The reference is optional and is ignored by this plotter. The method values remain the saved plug-in estimates; no confidence correction is added by plotting.

On your configured cluster, `jobs.sbatch` now runs all these representations and forwards additional arguments. For example:

```bash
sbatch jobs.sbatch --results_dir results/iw_all_representations --seeds 0
```

The job's existing environment, paths, and resource configuration are retained. Eight representations cost substantially more than a logits-only run; no training has been submitted automatically. To limit an exploratory run, pass dataset/method/threshold filters to that job. To retain the previous behavior, pass `--bound_strategies logits`.

`PCA<N>` is an uncentered SVD dimension **divisor**, not an output dimension. PCA1 uses all available SVD columns, which can be fewer than the full feature dimension if the fitting split is small. `features` is the unprojected feature array. All plots use the same fixed classifier outputs; these are different estimator representations, not retrained task classifiers.

The two committed hybrid result sets currently contain only logits. They cannot reproduce feature/PCA panels without another evaluation run. Do not rename or duplicate logits rows as PCA results.

## 2. Generate the other baseline predictions

The original accuracy comparison requires a separate `other_methods.pkl` file. It is not included in the existing committed results. Generate it once from the **same feature data**:

```bash
python -m pip install POT
mkdir -p results/baselines
python -m src.eval.other_methods \
  --feats_dir data/features --results_dir results/baselines
```

`POT` provides the `ot` dependency used by COT; install it only if it is missing from your environment. The existing baseline runner generates ATC, ATC_NE, COT, AC, and DOC with both source-calibrated and uncalibrated confidence. These baseline results use the historical baseline evaluation protocol; hybrid fitting uses its separate split protocol. The plotter matches dataset/shift/frozen-training-method and checks that reported full-target accuracies agree.

## 3. Produce all requested figures

```bash
python -m src.plot.iw_paper_figures \
  --results results/iw_all_representations/iw_hybrid_*.pkl \
  --other_results_fname results/baselines/other_methods.pkl \
  --plot_dir results/iw_paper_figures
```

The default runs both methods, all saved thresholds/seeds, both DA groups, and all three figure types. For one method and setting:

```bash
python -m src.plot.iw_paper_figures \
  --results results/iw_all_representations/iw_hybrid_*.pkl \
  --other_results_fname results/baselines/other_methods.pkl \
  --methods iw_overlap_critic --iw_thresholds 1 --seeds 0 \
  --da no --plot_dir results/iw_paper_overlap_w1
```

Use `--methods iw_residual_dis2` for the second method. Threshold and seed are never chosen by maximizing predicted or actual accuracy. Overlapping runs with duplicate experiment keys are rejected: pass only the intended run files, rather than a wildcard over several reruns.

The original comparison uses ATC_NE, COT, and AC with source temperature calibration. To include **every baseline implemented by the baseline runner**, add:

```text
--baseline_methods ATC ATC_NE COT AC DOC
```

Use `--temperature none` for uncalibrated baselines. `--bound_strategy PCA4`, for example, changes the hybrid representation in the accuracy-versus-baselines plot; it does not change the logits/full-feature and all-PCA panels.

## Use existing logits results immediately

After generating or supplying the separate baseline file, the accuracy comparison does not need another hybrid run:

```bash
python -m src.plot.iw_paper_figures \
  --results results/iw_hybrid_f475b82dfdbf.csv \
  --other_results_fname results/baselines/other_methods.pkl \
  --figures accuracy --methods iw_overlap_critic --iw_thresholds 1 \
  --plot_dir results/iw_accuracy_only
```

Flat CSV and pickle inputs are both accepted. CSV avoids NumPy/pandas pickle-version dependencies. If you only want representation figures, the baseline file is unnecessary:

```bash
python -m src.plot.iw_paper_figures \
  --results results/iw_all_representations/iw_hybrid_*.pkl \
  --figures representations --plot_dir results/iw_representations
```

By default, missing representations produce an actionable error before rendering. `--allow_missing` explicitly permits incomplete representation figures, labeling absent panels **Not run**. It does not manufacture measurements.

## Reading and checking the output

Each filename starts with method, threshold, seed, DA group, and prediction-column choice. A complete setting produces three PNGs, three PDFs, a `*_points.csv`, and a `*_summary.csv`.

- The default y-values are the saved `lower_bound` column. For hybrids these are clipped plug-in accuracy estimates, not certified lower bounds. `--prediction_column accuracy_estimate_raw` uses the unmodified raw accuracy estimates instead. The axis expands to display out-of-range estimates.
- Baselines always use their saved `lower_bound` predictions. There is no DIS² formula reconstruction, artificial `max_ts_agree_diff`, or added epsilon.
- The comparison restricts all requested series to common dataset/shift/training-method keys. The summary reports `n_available`, `n_matched`, `n_plotted`, and `n_unsupported_or_nonfinite` so exclusions are visible.
- Representation panels retain the available rows for each representation; panel counts expose missing or unsupported cases. They do not select the most favorable representation per experiment.
- Unsupported/nonfinite estimates are omitted from the scatter and counted. Empirical violations remain plotted and counted in coverage; they are not treated as unsupported.
- `*_points.csv` records the plotted prediction, actual accuracy, method, panel, identifiers, and whether the row was plotted. `*_summary.csv` gives MAE, empirical coverage, and mean violation magnitude for each series/panel.
- Target accuracy on the x-axis uses all supplied target samples, matching the original figure convention. Different full-target accuracies for matched baseline/hybrid keys cause an error instead of a misleading comparison.

There is no hybrid counterpart of the original "without delta" versus "with delta" panel: current hybrid estimates have no formal confidence correction. These requested figures do not require the historical `min_ratio` diagnostic.

## Validation

The plotting tests use only NumPy, pandas, and Matplotlib, without PyTorch or feature files:

```bash
python -m unittest discover -s tests -p test_iw_paper_figures.py -v
```

They check saved-value preservation, method/threshold/seed isolation, duplicate rejection, matched baseline cases, target-accuracy consistency, missing-input handling, all PCA panels, unsupported counts, and actual PNG/PDF/CSV generation. Full real-data PCA and baseline reproduction still requires the input runs above.
