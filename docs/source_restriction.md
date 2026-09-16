# Source restriction: pruning and source-weighted critic training

This branch adds a separate `source_restriction` experiment family. It implements the two requested variants without changing the historical DIS² runner or the IW hybrids.

| Variant | Critic source data | Critic target data | Final source evaluation | Final target evaluation |
|---|---|---|---|---|
| `source_pruning` | Source samples with target probability at least tau; uniform loss | Full target fitting split; uniform loss | Retained source evaluation samples; unweighted | Full target evaluation split; unweighted |
| `source_weighted_critic` | All source fitting samples; agreement loss weighted by target probability | Full target fitting split; uniform loss | Full source evaluation split; unweighted | Full target evaluation split; unweighted |
| `source_restriction_dis2_reference` | All source fitting samples; uniform loss | Full target fitting split; uniform loss | Full source evaluation split; unweighted | Full target evaluation split; unweighted |

“Full target” always means **no region restriction within the relevant split**. Training still does not see held-out target evaluation samples. There is no target pruning, target-loss discounting, overlap IW risk term, or residual target-mass multiplication in either method.

## 1. Shared discriminator

We reuse [domain.py](../src/lib/domain.py): a linear domain classifier on frozen representations, standardized using fitting samples only, with temperature calibration on internal validation data. Source receives domain label 0 and target label 1. Equal per-domain mean losses give effective domain priors of 1/2 despite unequal sample counts.

Let

$$
q(x)=\widehat P(D=T\mid x).
$$

A larger q means more target-like according to the fitted discriminator. Both methods use **q itself**, in [0,1], not density-ratio odds q/(1-q). This bounded score provides a soft source weight and a threshold ranking. The code computes it as the sigmoid of calibrated log odds.

Neither a high q nor a probability near 0.5 proves distributional overlap or equal task-label conditionals. A misspecified or overfitted discriminator can give misleading rankings.

## 2. Hard restriction: `source_pruning`

For a fixed probability threshold tau, define

$$
R_\tau=\{x:q(x)\ge\tau\}.
$$

The same frozen rule is applied to the source fitting, validation, and evaluation splits. The inequality is **greater than or equal**: this retains source samples that look most like target. No top-k selection or automatic threshold tuning is performed. Higher tau usually retains fewer source samples. Tau=0 retains every source sample and supplies a useful regression check.

Train a linear critic g to agree with the frozen classifier h on the retained source fitting samples and disagree on all target fitting samples, using the ordinary DIS² losses. Select the actual epoch/repeat snapshot maximizing

$$
\widehat d_{T,\mathrm{val}}(g)-\widehat d_{S\mid R_\tau,\mathrm{val}}(g).
$$

Final evaluation uses

$$
\widehat U_{\mathrm{prune}}
=\widehat e_{S\mid R_\tau}
+\widehat d_T(g)-\widehat d_{S\mid R_\tau}(g)
+\varepsilon(n_{S,R},n_T,\delta),
$$

where

$$
\varepsilon(n_S,n_T,\delta)
=\sqrt{\frac{(n_S+4n_T)\log(1/\delta)}{2n_Sn_T}}.
$$

Here n_{S,R} is the retained **source evaluation count**, and n_T is the full target evaluation count. Source error and source disagreement are conditional means with denominator n_{S,R}. The correction must use this reduced source count. **Do not multiply the result by the retained source fraction or by a target-region mass.** This DIS² expression concerns the full target distribution, with source distribution changed to S conditional on R.

The source-error term is recomputed on the pruned source; the full-source source error is not substituted.

## 3. Soft restriction: `source_weighted_critic`

Keep every source sample. During critic fitting, minimize the per-critic objective

$$
\mathcal L(g)=\frac12\left[
\frac{\sum_{i\in S_{\mathrm{fit}}}q(x_i)\ell_{\mathrm{CE}}(g(x_i),h(x_i))}
{\sum_{i\in S_{\mathrm{fit}}}q(x_i)}
+\frac{1}{n_{T,\mathrm{fit}}}\sum_{j\in T_{\mathrm{fit}}}\ell_{\mathrm{dis}}(g(x_j),h(x_j))
\right].
$$

Source weights are normalized by their sum. This reduces the **relative** source-agreement penalty on less target-like source samples while preserving total source-loss strength. It is not the same as shrinking the whole source loss by the mean q. Target weights remain uniform. Source pseudo-labels are the frozen classifier predictions, not source task labels.

“Finding the critic” includes model selection: on internal validation samples, select epoch and repeat by

$$
\widehat d_{T,\mathrm{val}}(g)
-\frac{\sum_{i\in S_{\mathrm{val}}}q(x_i)\mathbf1\{g(x_i)\ne h(x_i)\}}
{\sum_{i\in S_{\mathrm{val}}}q(x_i)}.
$$

Once this critic is selected, **discard the weights for final risk accounting**:

$$
\widehat U_{\mathrm{weighted\ critic}}
=\widehat e_S+\widehat d_T(g)-\widehat d_S(g)
+\varepsilon(n_S,n_T,\delta).
$$

All three means are unweighted. The source and target counts are the full final-evaluation split sizes. There is no weighted source-error estimate, weighted final discrepancy, or effective-sample-size substitution in the correction. The source-weight ESS is saved only as a training diagnostic.

There is no threshold in this variant: it is run once per seed/representation, not once per requested pruning threshold. `source_threshold` is NaN for this variant and the reference.

## 4. Splits and reference

The default is two outer blocks: 50% development and 50% untouched final evaluation. Development is internally divided as follows:

| Role | Fraction of all source/target samples | Reuse |
|---|---:|---|
| Fitting | 40% | Discriminator fit, optional source SVD fit, critic fit |
| Internal validation | 10% | Discriminator temperature fitting, then critic epoch/repeat selection |
| Final evaluation | Remaining 50% | Source error, disagreements, and correction counts |

Source and target are shuffled separately using seed and seed+1. The discriminator is fitted/calibrated first, then frozen. Reusing fitting and internal validation samples creates dependence inside development; neither stage sees final evaluation data. This is intentionally more data-efficient than the earlier five-way IW split. It is not a new theorem or proof of critic dominance.

`--split_fractions .4 .1` sets the first two fractions; the remainder is evaluation. Unlike the IW entry point, this flag takes **two values**, not four. Fractions round down, and evaluation receives the remainder. Full splits need at least two samples. `--min_source_samples` (default 2) additionally checks retained source fit/validation/evaluation counts.

`--include_dis2_reference` adds full-source, unweighted DIS² with these same splits, representations, optimizer, and critic initialization. This is the controlled baseline for these experiments. It differs from both the historical runner and the earlier five-split reference. At tau=0, source pruning matches this new reference numerically under the same settings; constant soft weights also reduce to uniform source weighting.

The reused critic loop processes all source and target samples in chunks and takes one optimizer step per epoch. Its epoch budget is not equivalent to the original minibatch DIS² epoch budget. The selected snapshot is the actual selected epoch and repeat, not the final training epoch.

## 5. What the bound means

Both variants report a **DIS²-form candidate bound with the original statistical correction**, conditional on the relevant critic-dominance assumption. The hard variant requires that assumption for the pair (S restricted to R, T); the soft variant requires it for the original pair (S,T) and the critic found by the weighted procedure.

Pruning changes the source distribution, and weighted training changes critic optimization. Neither operation automatically preserves the original assumption. A correction controls sampling uncertainty under appropriate assumptions; it cannot repair a critic that misses the target error gap. `confidence_bound_available=True` records that a DIS² correction is available for a supported row, not that the assumption has been verified. `guarantee_status=conditional_on_dis2_critic_dominance` makes this explicit.

Unlike the earlier IW-risk hybrids, these methods do not introduce an estimated importance-weighted risk term or target regional mass into the final expression. That is why the final correction has the ordinary DIS² form rather than a new IW-risk concentration term. Still, no empirical or theoretical improvement is claimed before validation.

Potential benefits and costs:

- Hard restriction removes source-only constraints that could hinder a useful target critic, but increases sampling uncertainty and may leave too few source samples.
- Soft restriction relaxes those constraints smoothly without reducing final source evaluation counts. However, extra disagreement on low-weight source samples is still subtracted in the **unweighted** final discrepancy, potentially making the estimate optimistic. Inspect coverage, not only tightness.
- Domain probabilities express relative domain evidence, not support density. Extreme tau can retain atypical source outliers. Feature compression can hide relevant shift, and a linear domain classifier can miss nonlinear differences.
- Splitting remains sample-level rather than patient/group-aware. Correlated data need group-aware splits before a statistical interpretation.

## 6. Run the methods

The existing feature layout and dependencies are reused:

```bash
python -m pip install -r requirements-hybrid.txt
python -m src.eval.source_restriction \
  --feats_dir data/features --results_dir results/source_restriction \
  --datasets cifar10 --shifts 1 --train_methods ERM-aug-imagenet \
  --methods source_pruning source_weighted_critic \
  --source_thresholds 0 0.5 0.7 0.9 \
  --bound_strategies logits --seeds 0 1 2 \
  --include_dis2_reference
```

Four pruning thresholds + one weighted variant + one reference yield six rows per dataset/shift/representation/seed, including unsupported rows. Omit dataset/shift/training-method filters to cover configured combinations. Missing feature files are reported and skipped.

For logits, full features, and all PCA figures:

```bash
python -m src.eval.source_restriction \
  --feats_dir data/features --results_dir results/source_restriction_all \
  --bound_strategies logits features PCA1 PCA4 PCA16 PCA32 PCA64 PCA128 \
  --source_thresholds 0 0.5 0.7 0.9 --seeds 0 \
  --include_dis2_reference
```

A separate cluster script preserves the existing IW job:

```bash
sbatch jobs_source_restriction.sbatch --source_thresholds 0 0.5 0.7 0.9 --seeds 0
```

That job uses the existing cluster environment/resource settings, defaults to all eight representations, and accepts extra CLI arguments. It has not been submitted automatically.

Important defaults:

| Option | Default | Meaning |
|---|---|---|
| `--methods` | Both variants | Restriction methods |
| `--source_thresholds` | `.5 .7 .9` | Probability cutoffs for hard pruning only; range [0,1] |
| `--split_fractions` | `.4 .1` | Fitting and internal validation; rest evaluation |
| `--bound_strategies` | `logits` | Also `features`, `PCA1`, `PCA4`, etc. |
| `--seeds` | `0` | Split/training repetitions |
| `--domain_epochs` | `100` | Discriminator optimizer steps |
| `--epochs` | `50` | Critic optimizer steps |
| `--critic_repeats` | `30` | Critic initializations selected on internal validation |
| `--batch_size` | `256` | Chunk size |
| `--loss_type` | `disagreement` | Also `DBAT`, `negative_xent` |
| `--min_source_samples` | `2` | Minimum retained source count in each split |
| `--delta` | `.01` | DIS² correction probability parameter |
| `--device`, `--threads` | `auto`, `1` | Runtime controls |

There is no `--iw_thresholds` or `--source_strength` option in this family. Source loss strength is fixed at the ordinary DIS² relative coefficient; soft weighting only redistributes it across source samples.

The benchmark CLI currently requires target labels in `ytrue.npy` for reporting, but the core estimator has **no target-label argument**. Target labels are attached only after predictions are computed. The loader, fixed head, source labels, and original feature layout are unchanged.

## 7. Results and plots

The CLI writes a DataFrame as `source_restriction_<run_id>.pkl`, a flat CSV, and JSON configuration. It also saves, per experiment/representation/seed:

- `source_restriction_splits_*.npz`: exact source/target indices for fit, validation, and evaluation.
- `source_restriction_scores_*.npz`: source target-probabilities aligned with the original source feature order, plus one hard mask per pruning threshold.
- `source_restriction_models_*.pth`: discriminator state and selected critic state dictionaries, keyed by `model_key`, with representation metadata. PCA models expect projected inputs; reconstruct the source-fitted projection from the saved fitting indices and original features when restoring them.

Run IDs hash configuration, not data or code. Rerunning the same configuration overwrites its outputs. Save code revision/environment separately and use separate result directories for different code/data revisions.

| Field | Meaning |
|---|---|
| `prediction_method`, `source_threshold`, `seed`, `bound_strategy` | Method configuration; soft/reference threshold is NaN |
| `source_error`, `source_disagreement`, `target_disagreement` | Final **unweighted** means on the appropriate source and full target evaluation sets |
| `max_ts_agree_diff` | Final selected critic's target minus source disagreement; legacy field name, not an evaluation-set maximization |
| `n_val_source`, `n_val_target` | Actual counts used in the final correction |
| `n_source_eval_full`, `n_target_eval_full` | Counts before source pruning |
| `source_retained_fraction` | Retained source evaluation fraction; diagnostic only |
| `n_source_fit_used`, `n_source_validation_used` | Source counts used to find the critic |
| `source_fit_effective_n` | ESS of training weights; not used in the correction |
| `error_estimate_raw`, `accuracy_estimate_raw` | Uncorrected empirical DIS² expression and its accuracy complement |
| `epsilon`, `error_upper_bound_raw`, `accuracy_lower_bound_raw` | Correction and corrected expressions before display clipping |
| `lower_bound` | Corrected accuracy expression clipped to [0,1] |
| `trg_accuracy`, `trg_eval_accuracy` | Actual full-target and evaluation-only accuracy for benchmark reporting |
| `bound_valid`, `bound_valid_eval` | Empirical coverage indicators, not verification of the population assumption |
| `status` | `ok`, `unsupported_source_eval/fit/validation`, or `unsupported_source_weights` |

Unsupported cases produce NaN predictions and no coverage classification; they are not silently assigned zero risk. Counts, scores, and retained fraction are still saved.

Compare both variants with the matching full-source reference:

```bash
python -m src.plot.source_restriction \
  --results results/source_restriction/source_restriction_*.pkl \
  --plot_dir results/source_restriction_figures
```

Create logits/full-feature and all-PCA panels:

```bash
python -m src.plot.source_restriction \
  --results results/source_restriction_all/source_restriction_*.pkl \
  --figures comparison representations --plot_dir results/source_restriction_all_figures
```

For the original accuracy-vs-ATC/COT/AC figure with a source-restriction method:

```bash
python -m src.plot.source_restriction \
  --results results/source_restriction/source_restriction_*.pkl \
  --methods source_pruning --source_thresholds 0.5 \
  --figures accuracy --other_results_fname results/baselines/other_methods.pkl \
  --plot_dir results/source_pruning_vs_baselines
```

See [baseline generation instructions](iw_paper_figures.md) if `other_methods.pkl` is missing. Plot inputs can be pickle or flat CSV. Plots read saved values, retain all selected settings separately, and save points/summary CSVs. No fake IW-threshold field is introduced. `--prediction_column accuracy_estimate_raw` compares uncorrected expressions; default `lower_bound` uses the correction. `--da yes/no/both`, `--seeds`, `--methods`, and `--source_thresholds` filter results. A threshold filter retains the threshold-free soft/reference rows unless methods are also filtered. Missing representation panels fail clearly unless `--allow_missing` is given.

## 8. Validation and recommended experiments

```bash
python -m unittest discover -s tests -p test_source_restriction.py -v
```

Tests cover pruning direction, split independence, correct source counts and correction, tau=0 equivalence to the reference, constant-weight equivalence, pruning of all source splits with full target retention, source weighting during fitting/selection only, unweighted final risk accounting, empty-source status, a real-discriminator end-to-end run through all representations, target-label isolation, checkpoint/score artifacts, and plot generation.

Start with tau=0 to verify the reference reduction, then predeclare a threshold sweep and measure retained counts, coverage, violation magnitude, and excess above true error on identical seeds/representations. Compare corrected and uncorrected outputs separately. Do not choose thresholds from final target task labels. Real-data improvement over DIS² or ODD has not yet been established.

Implementation: [core](../src/lib/source_restriction.py), [CLI](../src/eval/source_restriction.py), [plots](../src/plot/source_restriction.py), [tests](../tests/test_source_restriction.py). Underlying DIS² assumptions and correction: [Rosenfeld and Garg](https://arxiv.org/abs/2306.00312).
