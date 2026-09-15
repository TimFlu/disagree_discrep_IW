# Importance-weighted hybrid estimators

This branch adds `iw_overlap_critic` and `iw_residual_dis2`. The historical
`src.eval.dis2` runner, its losses, validation functions, and plots are unchanged.
The hybrids are **plug-in empirical estimates**, not established high-probability
bounds. All target task labels are used only for benchmark reporting.

## Shared region and importance weighting

A linear domain discriminator is fitted on frozen representations. Its features
are standardized using domain-training samples only. Source and target binary
cross-entropy losses each receive weight 1/2, irrespective of their sample counts.
Thus its effective sampling priors are equal. Temperature is fitted separately
on domain-calibration data, again using equal per-domain losses.

For target-domain posterior d(x), the general density ratio is
`w(x) = (pi_S / pi_T) * d(x) / (1-d(x))`. The implemented discriminator has
`pi_S = pi_T = 1/2`. It computes weights from calibrated log odds rather than
rounded sigmoid probabilities. Log odds are numerically limited to [-60, 60];
the number affected is reported. Thresholds are limited to (0, 1e6], so very
large ratios remain outside A rather than being clipped into the IW region.
There is no statistical weight clipping or self-normalization of the IW risk.

For each fixed threshold W, define `A = {x: w_hat(x) <= W}`, `B = A^c` using
exactly the same rule in both domains. This selects bounded estimated ratios;
it does NOT establish support, calibration, or equality of label conditionals.
The rule is based on estimated ratios, not an absolute-density overlap test.

The common overlap term on the final source evaluation split is

```
R_A = sum_i [1_A(x_i) * w_hat(x_i) * 1(h(x_i) != y_i)] / n_source_eval
m_B = n_target_eval_B / n_target_eval
```

The full source evaluation denominator is essential. Region counts are not
substituted for it. Regional disagreements below are conditional means.

## Method 1: iw_overlap_critic

Train a linear critic using source A and target B. Source labels are the deployed
classifier's predictions (not ground truth). Source agreement cross-entropy is
weighted by w_hat and normalized by the sum of weights WITHIN source A. Target
B receives the conditional mean disagreement surrogate. With source strength
lambda, the minimized loss is `(lambda * source_mean + target_mean) / 2`.
This explicitly fixes normalization; lambda is relative to the conditional
region objectives and not to their full-domain probability masses.

Select the repeat and epoch maximizing
`d_target_B - lambda * weighted_d_source_A` on critic-selection data.
The stored parameters are the parameters at that actual repeat AND epoch.
Final empirical error expression:

```
error_estimate_raw = R_A + m_B * d_target_B
```

An upper-bound interpretation requires conditional invariance on A and a
regional critic-dominance assumption on B. Agreement on source A does not
prove this assumption. A ground-truth-constrained critic is a future ablation,
not included in this implementation.

## Method 2: iw_residual_dis2

Train a fresh linear critic on source B and target B using ordinary unweighted
DIS2 source-agreement and target-disagreement losses. Select by
`d_target_B - d_source_B` on independent selection samples. The source strength
is always 1 for this method. Final empirical error expression:

```
error_estimate_raw = R_A + m_B * (error_source_B + d_target_B - d_source_B)
```

An upper-bound interpretation requires conditional invariance on A and the
DIS2 assumptions for the conditional residual distribution pair. Those
assumptions do not automatically transfer from the full domains.

Neither method is guaranteed to be tighter than the other.

## Splits, batching, and representations

For each seed, source and target are independently shuffled into disjoint splits:

| Split | Default fraction | Use |
| --- | ---: | --- |
| Domain training | 0.20 | Standardization and discriminator fitting |
| Domain calibration | 0.10 | Temperature fitting |
| Critic training | 0.30 | Critic optimization |
| Critic selection | 0.15 | Repeat/epoch selection |
| Final evaluation | Remainder, about 0.25 | IW, residual errors/disagreements, region mass |

Integer rounding is applied to the first four sizes; evaluation receives the
remainder. Each split needs at least two samples. Exact split indices are saved
as NPZ files. `--split_fractions` changes the first four fractions. No threshold
is selected automatically: every requested threshold is retained as its own row.
Do not select a threshold using benchmark target labels.

By default the representation is `logits`. `features` and `PCA<N>` are supported;
PCA<N> uses an uncentered SVD projection with approximately input_dim/N columns,
limited by the available rank, fitted on SOURCE DOMAIN-TRAINING data only.
For each representation the domain classifier and critics use that same space.
Conditional invariance is required in that space; reducing dimension does not
establish it. Frozen deployed predictions do not change with representation.

Each critic epoch accumulates gradients over every sample of each domain in
chunks, then takes one optimizer step. The two domains have separate
normalizers and neither is truncated to the other's loader length. Repeats
are optimized together. Epochs therefore count full-objective optimizer steps,
not the historical runner's minibatch steps; tune training length accordingly.

## Optional comparison reference

`--include_dis2_reference` adds full-domain DIS2 using the SAME critic training,
selection, and evaluation splits and the new checkpoint-correct training loop.
It is named `dis2_reference`, not substituted for historical results. Its saved
`lower_bound` includes the original correction

```
epsilon = sqrt((nS_eval + 4*nT_eval) * log(1/delta) / (2*nS_eval*nT_eval))
```

This retains DIS2's assumptions. Hybrid `epsilon` is NaN and
`confidence_bound_available=False`; adding just the residual DIS2 correction
would not produce a justified confidence bound for the complete hybrid.
A complete analysis must cover fitted weights, adaptive selection, IW
uncertainty, residual sample counts, and target region mass estimation.

## Running

Install dependencies into your environment:

```bash
python -m pip install -r requirements-hybrid.txt
```

Example for one dataset/shift; use the repository's existing extracted feature
format (`model_feats.npy`, `ytrue.npy`, `linear.pth`):

```bash
python -m src.eval.iw_hybrid \
  --feats_dir /path/to/features --results_dir results/iw_hybrids \
  --datasets cifar10 --shifts 1 --train_methods ERM-aug-imagenet \
  --methods iw_overlap_critic iw_residual_dis2 \
  --bound_strategies logits --iw_thresholds 1 2 5 \
  --seeds 0 1 2 --epochs 100 --domain_epochs 100 --critic_repeats 30 \
  --include_dis2_reference
```

Omit dataset, shift, and training-method filters to evaluate all configured
combinations; missing feature directories are reported and skipped. Use
`--device cpu` if needed. Each exact CLI configuration has a deterministic
run identifier; rerunning it overwrites its outputs. Use another results
directory if you need to retain different code versions of the same run.

## Outputs and compatibility

The runner writes `iw_hybrid_<run_id>.pkl` (pandas DataFrame), a CSV without
nested critic histories, JSON configuration, and NPZ split indices. Rows are
keyed by dataset, shift, train_method, bound_strategy, prediction_method,
iw_threshold, and seed. They retain common DIS2 fields such as `lower_bound`,
`trg_accuracy`, `src_accuracy`, `h_val_acc`, `h_full_acc`, `n_val_source`,
`n_val_target`, and `bound_valid`. There is no invented `max_ts_agree_diff`
for method 1, whose formula contains no such discrepancy.

`accuracy_estimate_raw = 1 - error_estimate_raw` is always retained.
For hybrid display compatibility, `lower_bound` clips that value to [0, 1].
It is not a certified bound. Out-of-range raw estimates are flagged rather
than erased. For the reference, `lower_bound` additionally subtracts epsilon.

As in the original runner, `trg_accuracy` uses all supplied target labels.
`trg_eval_accuracy` uses only the held-out final evaluation split.
`bound_valid` compares the saved lower_bound to full target accuracy;
`bound_valid_eval` compares to evaluation-only accuracy. Neither boolean is
proof of population coverage. Use the same evaluation population when
comparing runs; historical and new runs have different splitting protocols.

Additional diagnostics include IW/residual contributions, region counts,
selected weight sum, effective sample size, weight quantiles, calibration Brier
score, selected critic epoch/repeat/score, and numerical saturation count.
`n_source_a/b` and `n_target_a/b` refer to final evaluation samples.

If target A has samples but source A has none, IW is marked unsupported.
If residual DIS2 has target B but no evaluation source B, it is unsupported.
Empty training/selection regions are also unsupported, with NaN prediction
and no valid/invalid classification. If final target B is empty, the plug-in
expression reduces to IW; this observation does not prove population B is empty.
No silent fallback to zero risk is used for unsupported regions.

## Plots and validation

The historical plotter reconstructs DIS2's formula. Use the new plotter to
compare saved outputs directly:

```bash
python -m src.plot.compare_iw_hybrids \
  --results results/iw_hybrids/iw_hybrid_RUN_ID.pkl /path/to/historical_dis2.pkl \
  --bound_strategy logits --plot_dir results/iw_comparison
```

It produces PNG/PDF accuracy plots and a CSV summary with empirical coverage,
MAE, signed gap, excess error, violation magnitude, and unsupported counts.
Each threshold has its own series. Add `--DA` for DANN/CDANN; by default other
training methods are shown. No LaTeX installation is needed. Historical rows
are identified automatically; compare corrected DIS2 and plug-in hybrids
with their distinct statistical meanings in mind.

To recreate the basic paper-style comparisons for both IW methods, run:

```bash
python -m src.plot.iw_basic_figures \
  --results results/iw_hybrid_f475b82dfdbf.pkl results/dis2_50epochs_30repeats_valfrac0.50.pkl \
  --plot_dir results/iw_basic_figures
```

This writes PNG/PDF overlays, separate method panels, threshold comparisons,
and CSV summaries, separately for non-DA and DANN/CDANN models. Each method has
a distinct marker. W=1 is filled; larger thresholds use larger hollow markers.
All saved observations are retained without selecting the best repeat or W.
Only rows with status `ok` and finite prediction/target accuracy enter metrics;
excluded rows are counted in the summaries. Coverage is the empirical fraction
of saved estimates at or below target accuracy, including for plug-in methods.
Historical and reference rows can use different splits and experiment coverage;
these are descriptive comparisons, not paired tests.

Use `--methods dis2_historical dis2_reference iw_overlap_critic --iw_thresholds 1`
for the three-method W=1 comparison. `--bound_strategy` defaults to `logits`.
PCA, minimum-ratio, and alternative-loss ablations require corresponding saved
experiments/diagnostics; the current logits-only IW run cannot recreate them.
The historical delta correction is not applied to the IW plug-in estimates.

```bash
python -m pytest tests/test_iw_hybrid.py -q
```

Tests cover oracle mass-shift correction (10% source versus 60% target error),
regional denominators, both formulas and their same-critic difference,
empty-region behavior, prior correction, reproducible splits, saved checkpoint
selection, unequal-size batching, end-to-end feature-file evaluation and
plotting, and invariance of estimates to changed target task labels.
Real-data validation should next sweep known/estimated ratios, partial overlap,
conditional shift, threshold and sample size, reporting coverage and tightness
across seeds. This implementation does not establish that either hybrid is
valid on real-world shifts.
