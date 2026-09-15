# DIS² with importance-weighted regional estimators

This research fork extends the original [DIS² implementation](https://arxiv.org/abs/2306.00312) by Elan Rosenfeld and Saurabh Garg with two experimental methods for estimating a frozen classifier's error under distribution shift:

- **`iw_overlap_critic`**: estimate error in a selected region using importance weighting (IW), then use an overlap-constrained critic's disagreement on the remaining target samples.
- **`iw_residual_dis2`**: use the same IW estimate, then run a separate DIS² problem on the remaining source and target samples.

Both methods use labeled source data and target features without target task labels during estimation. They do not retrain or improve the deployed classifier. The intended improvement is in **performance estimation and, eventually, bound tightness**.

> **Current status:** both hybrids are implemented as **plug-in empirical estimates**, not complete high-probability bounds. The historical DIS² pipeline remains available. An optional `dis2_reference` uses the new pipeline's splits and training loop for a more controlled comparison. **ODD is a related external method, not an implemented baseline in this repository.** No real-data superiority over DIS² or ODD has been established here.

## Contents

- [Quick start](#quick-start)
- [What changed](#what-changed)
- [Methods and assumptions](#methods-and-assumptions)
- [Why improvement is plausible](#why-improvement-is-plausible)
- [Data and representations](#data-and-representations)
- [Running experiments](#running-experiments)
- [Results and evaluation](#results-and-evaluation)
- [Limitations and failure modes](#limitations-and-failure-modes)
- [Tests and research priorities](#tests-and-research-priorities)
- [Code map](#code-map)
- [Attribution and references](#attribution-and-references)

## Quick start

From a checkout of the `plan/iw-hybrid-methods` branch, run commands at the repository root:

```bash
python -m pip install -r requirements-hybrid.txt
python -m pytest tests/test_iw_hybrid.py -q

python -m src.eval.iw_hybrid \
  --feats_dir /path/to/features \
  --results_dir results/iw_hybrids \
  --datasets cifar10 --shifts 1 \
  --train_methods ERM-aug-imagenet \
  --methods iw_overlap_critic iw_residual_dis2 \
  --bound_strategies logits \
  --iw_thresholds 1 2 5 --seeds 0 1 2 \
  --include_dis2_reference

python -m src.plot.compare_iw_hybrids \
  --results results/iw_hybrids/iw_hybrid_*.pkl \
  --bound_strategy logits \
  --plot_dir results/iw_comparison
```

Replace the feature path with an existing extracted-feature dataset; the runner does not download or extract features automatically. Three seeds, three thresholds, and two hybrids produce **18 hybrid rows plus 3 reference rows**, including any unsupported rows. The reference is run once per seed/representation, not once per threshold.

Use a directory containing only the intended runs when passing the wildcard: the plotter concatenates files and does not deduplicate overlapping experiments.

## What changed

| Aspect | Historical DIS² | New hybrids / reference |
|---|---|---|
| Entry point | `src.eval.dis2` | `src.eval.iw_hybrid` |
| Domain classifier | None | Linear, temperature-calibrated discriminator for hybrids |
| Error accounting | Source error + full-domain disagreement discrepancy | IW contribution + method-specific residual contribution |
| Splits | Default 50% critic training / 50% validation | Domain fit, domain calibration, critic fit, critic selection, final evaluation |
| Critic selection | Maximum recorded validation discrepancy over epochs/repeats | Select epoch/repeat on a dedicated split; evaluate its actual parameter snapshot on untouched evaluation data |
| Optimization | Minibatch steps with paired source/target loaders | One full-objective step per epoch, processing both domains completely with separate normalizers |
| Statistical correction | Original DIS² correction | None for hybrids; original correction for `dis2_reference` |
| Results | Pandas `.pkl` | `.pkl`, flat `.csv`, configuration `.json`, split indices `.npz` |
| Plotting | Historical plots can reconstruct DIS²'s formula | New plotter reads saved predictions directly |

The original runner, losses, and validation code have not been replaced. The new critic loop reuses the original source loss and target-margin definitions. Splitting and optimization changes can themselves affect results: compare hybrids to **`dis2_reference` first**, and use historical DIS² results as a separate reproduction check.

The new loop retains the selected parameter snapshot **in memory**. It does not currently export trained discriminator or critic model files.

## Methods and assumptions

### Notation and prediction target

Let $S,T$ be source and target distributions, $\hat h$ the frozen deployed classifier, and $g$ a learned critic. In the formulas, $x$ denotes the estimator's representation. Frozen deployed predictions are computed once from the original head. The representation caveat below matters when this prediction cannot be reconstructed from a reduced representation.

| Symbol | Meaning |
|---|---|
| $e_D$ | Error of $\hat h$ under distribution $D$ |
| $d_D(g)$ | Disagreement $P_D(\hat h(X)\ne g(X))$ |
| $A$ | Region assigned to IW by a fixed estimated-ratio threshold |
| $B=A^c$ | Residual region |
| $S_B,T_B$ | Distributions conditional on $X\in B$ |
| $m_B$ | Target mass $P_T(B)$ |
| $R_A^{\mathrm{IW}}$ | Unnormalized target error contribution from $A$ |

Errors are in $[0,1]$. An error upper bound $U$ corresponds to an accuracy lower bound $1-U$. A larger accuracy lower bound is tighter **only while it remains valid**.

### 1. Historical DIS²

The critic is encouraged to agree with $\hat h$ on source samples and disagree on target samples. The population expression is

$$
U_{\mathrm{DIS^2}}=e_S+d_T(g)-d_S(g).
$$

Its upper-bound interpretation requires the paper's critic/discrepancy assumption. Optimizing a surrogate is not, by itself, proof that the condition holds. The empirical implementation adds

$$
\varepsilon(n_S,n_T,\delta)
=\sqrt{\frac{(n_S+4n_T)\log(1/\delta)}{2n_Sn_T}}.
$$

Source task labels measure source error. Critic training uses **the deployed classifier's predictions as pseudo-labels**, not source ground truth.

### 2. How ODD differs

[Overlap-aware Disagreement (ODD)](https://arxiv.org/abs/2506.14978) modifies critic training to reduce competing source-agreement and target-disagreement objectives in overlap. Its practical objective weights each target disagreement loss by the discriminator's target-domain probability, while retaining the source agreement loss.

The paper distinguishes a theoretical overlap set $D_\alpha$ from its practical soft domain-classifier proxy. Theorem 3.7 uses **non-overlap disagreement contributions**, source error, and a concentration term. Its assumptions include an ordering of target versus source disagreement with an ideal labeling hypothesis in overlap, plus a residual critic assumption. These regional contributions should not be confused with unweighted conditional means.

Our threshold $W$ is **not** the paper's density threshold $\alpha$. A domain posterior, a density ratio, and an absolute-density overlap set are different objects. ODD's target training weights should also not be assumed to be the weights in its final reported discrepancy without inspecting the particular ODD implementation.

The hybrids change **error accounting as well as critic training**: they explicitly estimate target error in A from weighted source errors. They are not merely another ODD loss switch. For an executable ODD comparison, run the [external ODD repository](https://github.com/aamixsh/odd) and document its exact revision, flags, and reported expression. That external baseline has not been reproduced here.

### 3. Shared discriminator, selection, and IW

A linear binary discriminator is trained on frozen representations, with source domain label 0 and target domain label 1. Features are standardized using domain-training data only. Source and target mean losses each receive weight $1/2$, even when their sample counts differ. Temperature scaling is fitted on the separate domain-calibration split using the same balance.

For a true domain posterior $d(x)=P(D=T\mid x)$ and effective training priors $\pi_S,\pi_T$,

$$
w(x)=\frac{p_T(x)}{p_S(x)}
=\frac{\pi_S}{\pi_T}\frac{d(x)}{1-d(x)}.
$$

The implemented effective priors are equal, so the prior factor is one. The fitted posterior gives an **estimated** ratio $\widehat w$. For each specified W,

$$
A=\{x:\widehat w(x)\le W\},\qquad B=A^c.
$$

The same rule is applied to source and target. Under equal priors, it selects $d(x)\le W/(1+W)$; for example, W=1 selects probabilities at most 0.5. This is a one-sided ratio rule, not a band around 0.5 or a certificate of overlap.

At the population level, with source support and equal label conditionals on A (and a frozen prediction determined by the representation),

$$
R_A^{\mathrm{IW}}
=\mathbb E_S[\mathbf1_A(X)w(X)\mathbf1\{\hat h(X)\ne Y\}]
=P_T(A)e_{T\mid A}.
$$

The final empirical estimator is

$$
\widehat R_A^{\mathrm{IW}}
=\frac{1}{n_S}\sum_{i=1}^{n_S}
\mathbf1_A(x_i)\widehat w(x_i)\mathbf1\{\hat h(x_i)\ne y_i\},
\qquad
\widehat m_B=\frac{n_{T,B}}{n_T}.
$$

Here $n_S,n_T$ are the **full final-evaluation split sizes**, before region selection. Do not divide the IW sum by $n_{S,A}$, normalize it by the weight sum, or multiply it by $P_T(A)$ again: it already estimates a regional contribution.

Ratios are computed from calibrated log odds, limited numerically to [-60,60]. Saturation counts are reported. Allowed thresholds are $0<W\le10^6$. There is no statistical weight-clipping option or self-normalization of the IW risk: large estimated ratios are assigned to B.

### 4. `iw_overlap_critic`: evidence in A, disagreement in B

Train a fresh linear critic using **source A and target B**. Its per-critic objective is

$$
\mathcal L_A(g)=\frac12\left[
\lambda\frac{\sum_{i\in S_A}\widehat w_i\,\ell_{\mathrm{CE}}(g(x_i),\hat h(x_i))}{\sum_{i\in S_A}\widehat w_i}
+\frac{1}{n_{T,B}}\sum_{j\in T_B}\ell_{\mathrm{dis}}(g(x_j),\hat h(x_j))
\right].
$$

These are critic-training samples, separate from final evaluation. Source loss is normalized **within A by its weight sum**; this differs intentionally from normalization of the final IW risk. Target loss is a conditional mean on B. `--source_strength` sets $\lambda$, default 1.

On the independent critic-selection split, choose the epoch and repeat maximizing

$$
d_{T_B}(g)-\lambda d_{S_A}^{\widehat w}(g),
$$

where source disagreement is a weight-normalized mean. Final error accounting is

$$
\widehat U_{\mathrm{overlap}}
=\widehat R_A^{\mathrm{IW}}+\widehat m_B\widehat d_{T_B}(g_A).
$$

Source disagreement constrains training and selection; it is **not subtracted in this final expression**. An upper-bound interpretation additionally requires

$$
e_{T_B}\le d_{T_B}(g_A).
$$

Agreement with source predictions in A does not establish this residual dominance condition. Both classifiers could agree and be wrong on B. Source-ground-truth-constrained critics are not implemented.

### 5. `iw_residual_dis2`: remove A and solve residual DIS²

Train a fresh linear critic using **source B and target B** with unweighted source agreement and target disagreement losses. The source coefficient is fixed at 1; `--source_strength` does not alter this method. Select the epoch/repeat maximizing $d_{T_B}(g)-d_{S_B}(g)$ on the critic-selection split.

The final expression is

$$
\widehat U_{\mathrm{residual}}
=\widehat R_A^{\mathrm{IW}}+
\widehat m_B\left[
\widehat e_{S_B}+\widehat d_{T_B}(g_B)-\widehat d_{S_B}(g_B)
\right].
$$

All quantities inside brackets are **conditional means over B**. The residual source-error term remains, and the **whole bracket** is multiplied by target residual mass. A bound requires the DIS² assumptions for the conditional pair $(S_B,T_B)$; validity for the full domains does not automatically imply validity after selection.

Example: IW contribution 0.06, target residual mass 0.30, and conditional residual DIS² expression 0.50 yield error estimate $0.06+0.30(0.50)=0.21$, hence raw accuracy estimate 0.79. This omits statistical uncertainty, as the implemented hybrid does.

### 6. Direct comparison

| Component | `iw_overlap_critic` | `iw_residual_dis2` |
|---|---|---|
| IW contribution | Same | Same |
| Source samples constraining critic | A | B |
| Source training weights | Estimated ratios, normalized within A | Uniform within B |
| Target critic samples | B | B |
| Final residual expression | Target disagreement | Source error + target disagreement − source disagreement |
| Additional assumption | Residual disagreement dominates error | Residual discrepancy dominates target–source error gap |
| Main weakness | Constraints in A may not control behavior in B | Residual source may be sparse or absent |

If both expressions used the **same** critic, their difference would be

$$
U_{\mathrm{residual}}-U_{\mathrm{overlap}}
=m_B[e_{S_B}-d_{S_B}(g)].
$$

The implementations learn different critics, so this identity is an algebraic diagnostic, not a guaranteed ordering of actual results. Neither method is automatically tighter.

## Why improvement is plausible

**Account for changed mass within shared support.** Shared support and unchanged labeling do not imply equal average error. Consider a classifier wrong throughout region C and correct throughout region D:

| Region | Source mass | Target mass | Error |
|---|---:|---:|---:|
| C | 0.10 | 0.60 | 1 |
| D | 0.90 | 0.40 | 0 |

Source error is 10%, target error 60%, despite identical label conditionals. Oracle ratios 6 and $4/9$ recover 60% if both regions are assigned to IW, for example with W≥6. A smaller threshold sends C to the residual method. This motivates measuring error in the selected region rather than inferring low error merely from overlap; it does not show that every ODD run fails on this example.

**Replace some conservatism with labeled evidence.** When ratios are accurate and source errors transfer on A, its target error contribution can be estimated directly. The critic handles a smaller target component. This can help where full-domain disagreement is conservative or its optimization suffers from overlap competition.

**Avoid relying on very large estimated IW weights.** Target-dominated regions can be assigned to the residual critic. Increasing W admits more samples to IW but may increase variance and reduce residual source counts. Decreasing W makes the result more dependent on the critic. There is no universally best W or monotone improvement guarantee.

**Separate two research questions.** The overlap critic tests whether reliable-region constraints control the remaining target region. Residual DIS² tests whether a familiar discrepancy argument remains useful after removing the IW region.

These are hypotheses to test. Accurate mass correction can also **increase** an error estimate when source performance was optimistic. A numerically smaller error expression is not, by itself, evidence of a better method.

## Data and representations

### Required feature format

The loader expects these paths beneath `--feats_dir`:

```text
cifar10/ERM-aug-imagenet_1_100.0/model_feats.npy
cifar10/ERM-aug-imagenet_1_100.0/ytrue.npy
cifar10/ERM-aug-imagenet_1_100.0/linear.pth
```

In general the folder is `<dataset>/<train_method>_<shift>_100.0/`.

| File | Content |
|---|---|
| `model_feats.npy` | Saved dictionary with `source` and `target` feature arrays, shapes `(nS,d)` and `(nT,d)` |
| `ytrue.npy` | Saved dictionary with aligned integer `source` and `target` label arrays |
| `linear.pth` | State dictionary of the frozen `torch.nn.Linear(d,K)` classifier, with weight and bias |

Use zero-based class indices. The loader infers K from `max(source_labels)+1`; ensure this matches the saved head and target label space. Logits are recomputed from features and the frozen head.

**Target-label isolation:** the core `evaluate_hybrids` accepts no target task labels. However, the current **benchmark CLI and shared loader still require target labels in `ytrue.npy`** to report actual accuracy. A truly unlabeled deployment should call the core estimator with features/logits; a dedicated label-optional CLI remains future work. Do not present fabricated labels as benchmark truth.

Original DIS² features are available from the authors' [Google Drive archive](https://drive.google.com/file/d/14Jh0C1MrBre0RebYSbEU4qrznDYMDyr_/view). The original README supplied SHA-256 `8a079e55cc03c6e2cd684dc705b5d79afe0c5e41f0010e8e869a7803695182ba`.

### Representation choices

| `--bound_strategies` | Meaning |
|---|---|
| `logits` | Frozen classifier logits; default |
| `features` | Original extracted features |
| `PCA1`, `PCA4`, `PCA16`, … | Uncentered SVD projection fitted only on source domain-training samples |

`PCA<N>` uses a **dimension divisor**, not a requested output dimension: nominally `max(1, input_dim // N)` columns, capped by available SVD columns. It is not centered PCA. Discriminator and critics use the same selected representation; frozen classifier predictions do not change.

Conditional invariance must hold for the error-relevant information in this space. For a reduced representation Z where the frozen prediction is not determined by Z, equality of $P(Y\mid Z)$ alone is insufficient: transferring conditional error also requires stability of the information determining the frozen prediction. Neither dimensionality reduction nor visual overlap establishes these conditions.

### Split protocol

Source and target are independently shuffled per seed (source uses `seed`, target `seed+1`). Methods and thresholds share these splits.

| Split | Default fraction | Purpose |
|---|---:|---|
| `domain_train` | 0.20 | Standardization, optional SVD, discriminator fitting |
| `domain_cal` | 0.10 | Discriminator temperature fitting |
| `critic_train` | 0.30 | Critic optimization |
| `critic_select` | 0.15 | Select critic epoch and repeat |
| `eval` | Remaining ≈0.25 | Final risk/disagreement estimates and target region mass |

The first four sizes are rounded down; evaluation receives the remainder. Every full split needs at least two samples. Selected regional subsets can still be empty. Splits are sample-level random splits, not class-stratified or patient/group-aware; grouped medical datasets need an adapted splitting protocol before use.

## Running experiments

### One method or a larger comparison

Use `--methods iw_overlap_critic` or `--methods iw_residual_dis2` to run one method. Omit the flag for both. A broader explicit configuration is:

```bash
python -m src.eval.iw_hybrid \
  --feats_dir /path/to/features --results_dir results/iw_study \
  --datasets cifar10 --shifts 1 10 71 95 \
  --train_methods ERM-aug-imagenet \
  --bound_strategies logits features PCA4 \
  --iw_thresholds 0.5 1 2 5 10 \
  --seeds 0 1 2 3 4 \
  --epochs 100 --domain_epochs 100 --critic_repeats 30 \
  --include_dis2_reference --device cpu
```

Omit dataset, shift, and training-method filters to iterate over combinations configured in [consts.py](src/lib/consts.py). Missing files are reported and skipped. `--shifts` filters configured shifts; it does not add new ones. Adding a dataset requires extending the configured dataset/shift map.

### CLI options

| Option | Default | Meaning |
|---|---|---|
| `--feats_dir`, `--results_dir` | Required | Feature root and output directory |
| `--methods` | Both hybrids | Methods to evaluate |
| `--iw_thresholds` | `1 2 5` | Fixed ratio thresholds; one row per threshold/method |
| `--datasets` | All configured | Dataset filter |
| `--shifts` | All configured | Integer shift filter |
| `--train_methods` | All configured | Frozen-model training-method filter |
| `--bound_strategies` | `logits` | Estimator representations |
| `--seeds` | `0` | Split/training seeds |
| `--epochs` | `50` | Critic full-objective optimizer steps |
| `--domain_epochs` | `100` | Discriminator full-objective optimizer steps |
| `--critic_repeats` | `30` | Parallel random critic initializations |
| `--batch_size` | `256` | Gradient-accumulation/evaluation chunk size |
| `--source_strength` | `1` | Source-A penalty for overlap critic only |
| `--loss_type` | `disagreement` | Also `DBAT` or `negative_xent` |
| `--split_fractions` | `.2 .1 .3 .15` | First four split fractions; remainder is evaluation |
| `--include_dis2_reference` | Off | DIS² with matching splits and new training loop |
| `--delta` | `.01` | Confidence parameter for DIS² reference only |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--threads` | `1` | PyTorch CPU threads |

Run `python -m src.eval.iw_hybrid --help` for the parser. The discriminator uses AdamW with learning rate 0.01 and weight decay 0.0001; critics use 0.003 and 0.0005. These are code defaults, not CLI options. Calibration uses LBFGS to fit temperature.

One new-loop epoch processes every sample in each selected domain and takes **one** optimizer step. Historical epochs contain multiple minibatch steps, so matching `--epochs` does not match optimization budgets. Repeats are random starts for critic selection, not independent benchmark replications; use seeds for repeated experiments.

### Historical DIS² reproduction

```bash
mkdir -p results/dis2_original
python -m src.eval.dis2 \
  --feats_dir /path/to/features \
  --results_dir results/dis2_original \
  --epochs 50 --critic_repeats 30 --val_frac 0.5
```

The historical CLI iterates over configured datasets, training methods, and representations. It does not accept the new runner's dataset, seed, or threshold filters. The example output filename is `dis2_50epochs_30repeats_valfrac0.50.pkl`.

## Results and evaluation

### Output files and reproducibility

| Output | Content |
|---|---|
| `iw_hybrid_<run_id>.pkl` | Full pandas DataFrame, including critic histories where available |
| `iw_hybrid_<run_id>.csv` | Flat results, excluding nested critic histories |
| `iw_hybrid_<run_id>.json` | CLI configuration and schema version |
| `splits_<dataset>_<train_method>_<shift>_<seed>_<run_id>.npz` | Exact source/target split indices |

`run_id` hashes CLI configuration. It does **not** hash code, feature contents, or dependency versions. Rerunning the same configuration overwrites its files. Save the code commit and environment separately; use a new output directory to preserve runs from different code/data revisions.

Rows identify `dataset`, `shift`, `train_method`, `bound_strategy`, `prediction_method`, `iw_threshold`, and `seed`. Reference rows have NaN thresholds.

### Which fields to analyze

| Field | Interpretation |
|---|---|
| `status` | `ok` or unsupported-region status; inspect first |
| `error_estimate_raw` | Empirical error expression before confidence correction |
| `accuracy_estimate_raw` | `1 - error_estimate_raw` |
| `lower_bound` | Clipped hybrid accuracy estimate, or corrected/clipped reference bound |
| `epsilon` | NaN for hybrids; DIS² correction for reference |
| `error_upper_bound_raw` | Reference only: raw error expression plus correction |
| `estimate_kind` | `plugin` or `dis2_with_correction` |
| `confidence_bound_available` | False for hybrids; true for reference under DIS² assumptions |
| `trg_accuracy` | Accuracy on **all supplied target labels** |
| `trg_eval_accuracy` | Accuracy on final target evaluation split |
| `bound_valid`, `bound_valid_eval` | Empirical comparison to full-target or evaluation-only accuracy |
| `iw_error_contribution` | IW contribution from source evaluation samples in A |
| `target_residual_mass` | Fraction of target evaluation samples in B |
| `residual_error_contribution` | Entire residual contribution, including target mass |
| `residual_source_error`, `residual_source_disagreement`, `residual_discrepancy` | Residual DIS² components, when evaluated |
| `residual_target_disagreement` | Conditional target-B disagreement, when evaluated |
| `n_source_a/b`, `n_target_a/b` | Final-evaluation regional counts |
| `iw_effective_n` | Selected-source weight ESS: $(\sum w)^2/\sum w^2$ |
| `iw_weight_sum`, `iw_weight_max`, `iw_weight_p50`, `iw_weight_p95` | Selected-source evaluation weight diagnostics |
| `domain_cal_brier`, `domain_cal_balanced_accuracy`, `domain_temperature` | Discriminator diagnostics on calibration data |
| `domain_log_odds_clipped_count` | Numerical saturation count across supplied source and target samples |
| `critic_epoch`, `critic_repeat`, `critic_selection_score` | Selected critic identifiers/score; epoch and repeat are zero-based |
| `output_clipped` | Hybrid raw estimate fell outside [0,1], where field is populated |

For hybrids, `lower_bound = clip(accuracy_estimate_raw, 0, 1)`. For reference, `lower_bound = clip(accuracy_estimate_raw - epsilon, 0, 1)`. The shared column name does **not** give hybrids a formal guarantee. Preserve raw values: clipping can hide negative error expressions, and estimated IW contributions can exceed one.

Brier score is measured on the split used to fit temperature, so it is a diagnostic, not an independent calibration guarantee. ESS measures weight concentration, not ratio correctness. The common DIS² fields `src_accuracy`, `h_full_acc`, `h_val_acc`, `n_val_source`, and `n_val_target` are retained. No artificial `max_ts_agree_diff` is provided for the overlap method: its expression has no source-subtracted discrepancy.

### Plots

```bash
python -m src.plot.compare_iw_hybrids \
  --results results/iw_hybrids/iw_hybrid_*.pkl \
            results/dis2_original/dis2_50epochs_30repeats_valfrac0.50.pkl \
  --bound_strategy logits --plot_dir results/iw_comparison
```

Outputs are `compare_iw_hybrids_logits.png`, `.pdf`, and `_summary.csv`. Each hybrid threshold is a separate series. Add `--DA` for DANN/CDANN rows; without it those two training methods are excluded. Repeat for other representations. No LaTeX installation is needed.

The x-axis is actual full-target accuracy and the y-axis is saved `lower_bound`. Points **above the diagonal overestimate accuracy** and are empirical violations; points below are conservative. The historical `compare_methods_acc_vs_pred` reconstructs a DIS²-specific expression and should not be used for hybrid rows.

The new plotter pools matching rows over datasets/shifts/seeds and uses full-target accuracy. It has no evaluation-only switch, group confidence intervals, or automatic ODD result conversion.

### Metrics and fair comparisons

For actual accuracy a and displayed prediction L, the plot summary reports:

| Metric | Definition / interpretation |
|---|---|
| Empirical coverage | Fraction with $L\le a$; higher is safer |
| MAE | Mean $\lvert a-L\rvert$; lower is closer, but ignores safety direction |
| Mean signed gap | Mean $a-L$; positive means conservative on average |
| Mean excess error | Mean $\max(a-L,0)$; zeros included for violations |
| Mean violation magnitude | Mean $\max(L-a,0)$; zeros included for valid rows |
| Unsupported count | Rows without an evaluable prediction |

Metrics use supported finite predictions; unsupported rows are counted separately. Do not silently discard unsupported experiments or report coverage without their count. A predictor returning zero accuracy is conservative but usually uninformative.

Use three clearly labeled comparisons:

1. **Displayed outputs:** hybrids versus corrected `dis2_reference`. This shows saved results, but any tightness advantage mixes method changes with omission of a confidence correction.
2. **Raw empirical expressions:** compare `accuracy_estimate_raw` for hybrids and reference. This removes the correction difference; all raw values remain empirical estimates.
3. **Historical reproduction:** compare historical DIS² separately, explaining differences in splitting, optimization, and selection.

Align dataset, shift, frozen training method, representation, and seed. At each threshold, compare methods on common supported cases **and** report overall support rates. Report dataset/shift-level results as well as aggregates. Seed repeats on one dataset are not independent new distribution shifts.

This standalone inspection uses held-out target evaluation accuracy and raw predictions consistently. Replace the filename with one actual result file:

```python
import numpy as np
import pandas as pd

df = pd.read_pickle("results/iw_hybrids/iw_hybrid_RUN_ID.pkl")
keys = ["prediction_method", "iw_threshold", "bound_strategy"]
rows = []
for key, group in df.groupby(keys, dropna=False):
    ok = group["status"].eq("ok") & np.isfinite(group["accuracy_estimate_raw"])
    supported = group.loc[ok]
    gap = supported["trg_eval_accuracy"] - supported["accuracy_estimate_raw"]
    rows.append(dict(zip(keys, key)) | {
        "n_total": len(group), "n_supported": int(ok.sum()),
        "raw_eval_coverage": float((gap >= 0).mean()),
        "raw_eval_mae": float(gap.abs().mean()),
        "raw_eval_violation_magnitude": float((-gap).clip(lower=0).mean()),
    })
print(pd.DataFrame(rows).to_string(index=False))
```

Target labels are for **post hoc benchmark evaluation only**. Do not use them to select W, representation, source strength, critic, or the winning seed. The runner retains every requested threshold; it does not choose one. A threshold chosen for best labeled-target results is an oracle analysis and must be labeled as such.

## Limitations and failure modes

| Issue | Consequence / current handling |
|---|---|
| Estimated ratios | A linear discriminator can miss nonlinear density differences; temperature scaling cannot repair arbitrary misspecification. Ratio errors affect weighting and selection. |
| Conditional shift | IW cannot correct different conditional errors. Representation overlap can conceal clinically important differences. |
| Incomplete hybrid confidence analysis | `--delta` affects only reference. A residual DIS² correction alone cannot cover IW uncertainty, estimated weights, or estimated region mass. |
| Critic assumption | Surrogate optimization and source agreement do not prove residual dominance. Required assumptions differ between the hybrids. |
| Small regional samples | Full splits can be adequate while A or B is too small. No minimum regional ESS/sample-size gate beyond nonemptiness checks. |
| Missing source A with target A present | `unsupported_iw_source_region`; prediction is NaN. |
| Missing evaluation source B with target B present | Residual method returns `unsupported_residual_source_region`. |
| Empty required critic training/selection region | `unsupported_critic_region`; no silent zero-error fallback. |
| No target B in final evaluation | Expression reduces to IW and critic fitting can be skipped. This does not prove population B is empty. |
| Empty A | Residual formula reduces to ordinary uncorrected DIS² if required B samples exist. Overlap critic cannot train without source A when target B remains. |
| Extreme W | More IW may increase variance or starve residual source; less IW increases dependence on the critic. |
| Sample-level splits | Correlation, duplicate subjects, or patient leakage undermine independence. Group-aware splitting is not implemented. |
| Data efficiency and compute | Five-way splitting reduces evaluation size; critics are trained per threshold/method/representation. Full arrays are loaded on the chosen device despite chunked optimization. |
| Incomplete artifacts | Configurations/splits are saved; trained models, per-sample weights/masks, dependency versions, and code/data hashes are not. |
| Benchmark-only CLI | Loader/reporting still require target labels, though core estimation excludes them. |
| No ODD runner | Requires an external, separately verified implementation and matched evaluation protocol. |

A complete high-probability analysis must control the IW contribution and weight-estimation error, the residual term, and estimation of target region mass, with suitable assumptions and probability allocation. Residual DIS² concentration would use **residual evaluation counts**, not full-domain counts. Independent fitting/evaluation helps with selection dependence but does not eliminate ratio bias or establish critic dominance. Selecting configurations after seeing final results also needs separate consideration.

## Tests and research priorities

```bash
python -m pytest tests/test_iw_hybrid.py -q
```

The suite contains 14 tests covering oracle IW mass correction (10% source versus 60% target error), denominators, residual target mass, both formulas and their same-critic difference, empty-region handling, raw out-of-range results, prior correction, reproducible disjoint splits, selected parameter snapshots, unequal-size batching, domain fitting, end-to-end files/plots, and invariance of predictions when target task labels change.

These are implementation and synthetic integration checks. They do not establish real-data performance, conditional invariance, density-ratio accuracy, or coverage at a claimed confidence level.

Suggested evaluation sequence:

1. **Oracle ratios and known regions:** isolate the decomposition from discriminator error; check true regional error recovery. Oracle-ratio experiments need a separate harness; they are not a CLI switch.
2. **Estimated ratios with shared support:** vary mass within support under fixed conditionals; measure bias, ESS, calibration, and support rates.
3. **Partial overlap:** vary target residual mass and source residual counts; inspect each term and diagnose critic assumptions using labels only for post hoc analysis.
4. **Assumption failures:** introduce conditional shift and representation compression. Report where estimates become optimistic, not only successful cases.
5. **Matched real-data comparisons:** hybrids, matching-split reference, historical DIS², and independently verified ODD. Report coverage, conservatism, violations, MAE, support rate, runtime, and seed variation.
6. **Ablations:** W, overlap-critic source strength, representation, sample size, optimization budget, discriminator capacity/calibration. Distinguish CLI options from extensions requiring code changes.

Useful next steps are a label-optional CLI, group-aware splits, model/per-sample diagnostic export, raw-versus-corrected and evaluation-only plot options, a controlled ODD baseline, and stronger ratio diagnostics. Ground-truth-constrained critics, cross-fitting, signed/positive-only corrections to an existing ODD estimate, and a complete finite-sample hybrid bound are **not implemented**.

The central question is whether reliable IW on A reduces conservatism enough to compensate for ratio error and a smaller, potentially harder residual problem. Answering it requires **coverage and tightness together**, not just higher predicted accuracy.

## Code map

| File | Responsibility |
|---|---|
| [src/eval/iw_hybrid.py](src/eval/iw_hybrid.py) | CLI, representations, experiment loop, outputs, benchmark labels |
| [src/lib/domain.py](src/lib/domain.py) | Discriminator, balanced fitting, calibration, ratios |
| [src/lib/overlap.py](src/lib/overlap.py) | Threshold masks, IW contribution, weight diagnostics |
| [src/lib/hybrid_critic.py](src/lib/hybrid_critic.py) | Optimization, independent selection, parameter snapshot |
| [src/lib/hybrid_validation.py](src/lib/hybrid_validation.py) | Splits, formulas, support checks, reference |
| [src/plot/compare_iw_hybrids.py](src/plot/compare_iw_hybrids.py) | Saved-output plots and summaries |
| [tests/test_iw_hybrid.py](tests/test_iw_hybrid.py) | Mathematical and end-to-end checks |
| [src/eval/dis2.py](src/eval/dis2.py) | Historical DIS² runner |
| [docs/iw_hybrids.md](docs/iw_hybrids.md) | Compact implementation notes |

## Attribution and references

This fork builds on the original code for **(Almost) Provable Error Bounds Under Distribution Shift via Disagreement Discrepancy** by Elan Rosenfeld and Saurabh Garg, NeurIPS 2023, volume 36. Claims in that paper about experimental validity refer to its experiments and assumptions, not guarantees for these new hybrids.

- [DIS² paper](https://arxiv.org/abs/2306.00312)
- [ODD paper, arXiv:2506.14978](https://arxiv.org/abs/2506.14978) — the comparison follows the supplied v1, particularly Section 3 / Theorem 3.7 and Section 4 / Algorithm 1.
- [External ODD code](https://github.com/aamixsh/odd)

Please cite the original work when using its method or code:

```bibtex
@inproceedings{rosenfeld2023almost,
    author    = {Elan Rosenfeld and Saurabh Garg},
    title     = {(Almost) Provable Error Bounds Under Distribution Shift
                 via Disagreement Discrepancy},
    booktitle = {Advances in Neural Information Processing Systems},
    volume    = {36},
    year      = {2023},
    url       = {https://arxiv.org/abs/2306.00312}
}
```
