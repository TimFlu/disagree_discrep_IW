# Historical DIS2 figures

Input: `results/dis2_50epochs_30repeats_valfrac0.50_compatible.pkl`.
This is a value-preserving copy of the original pickle, serialized using
NumPy 1.24.3 and pandas 2.0.3 (pickle protocol 4). String extension columns
were converted to ordinary object columns to remove newer pandas state and
loader-class dependencies. The original file is unchanged.

The five original plotting modules were run on this copy. They retain their
original non-DANN/CDANN filtering, DIS2 formula reconstruction, clipping,
and best-bound selection where implemented; these differ from the direct
saved-estimate comparisons in `iw_basic_figures`.

Reproduce each figure with:

```bash
for plot in accuracy_diff_vs_min_ratio dis2_decrease_min_ratio dis2_reduce_PCs dis2_variants dis2_vary_delta; do
  MPLBACKEND=Agg python -m src.plot.$plot \
    --dis2_results_fname results/dis2_50epochs_30repeats_valfrac0.50_compatible.pkl \
    --plot_dir results/dis2_figures
done
```

The plotting modules use Matplotlib mathtext when LaTeX is unavailable.
Baseline comparison/table generation requires a separate baseline-results
file. Alternative-loss comparison requires separate D-BAT/other-loss results.
Those inputs are not present in the results directory.
