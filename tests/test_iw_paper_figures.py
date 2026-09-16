"""Plot regressions run without PyTorch, feature files, or pytest."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.plot.iw_paper_figures import (REPRESENTATIONS, load_hybrids, load_baselines,
                                      matched_comparison, main)


def hybrid_fixture():
    rows = []
    for method in ('iw_overlap_critic', 'iw_residual_dis2'):
        for threshold in (1., 2.):
            for seed in (0, 1):
                for representation in REPRESENTATIONS:
                    for shift, train, truth in [(1, 'ERM-aug-imagenet', .7), (2, 'DANN', .6)]:
                        rows.append(dict(dataset='cifar10', shift=shift, train_method=train,
                            prediction_method=method, iw_threshold=threshold, seed=seed,
                            bound_strategy=representation, status='ok', trg_accuracy=truth,
                            lower_bound=.4 + .1*seed, accuracy_estimate_raw=.35 + .1*seed,
                            h_val_acc=.99, max_ts_agree_diff=123., epsilon=456.))
    return pd.DataFrame(rows)


def baseline_fixture():
    return pd.DataFrame([dict(dataset='cifar10', shift=shift, train_method=train,
        prediction_method=method, temperature='source', lower_bound=.55, trg_accuracy=truth)
        for method in ('ATC_NE', 'COT', 'AC')
        for shift, train, truth in [(1, 'ERM-aug-imagenet', .7), (2, 'DANN', .6)]])


class PaperFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hp, self.bp = self.root/'hybrid.csv', self.root/'baseline.pkl'
        hybrid_fixture().to_csv(self.hp, index=False)
        baseline_fixture().to_pickle(self.bp)

    def test_saved_values_and_method_threshold_seed_preserved(self):
        data = load_hybrids([self.hp], ['iw_residual_dis2'], [2.], [1])
        self.assertEqual(len(data), 2*len(REPRESENTATIONS))
        self.assertTrue((data.prediction == .5).all())
        self.assertEqual(set(data.bound_strategy), set(REPRESENTATIONS))
        raw = load_hybrids([self.hp], ['iw_residual_dis2'], [2.], [1], 'accuracy_estimate_raw')
        np.testing.assert_allclose(raw.prediction, .45)

    def test_duplicate_runs_are_not_silently_maximized(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate hybrid'):
            load_hybrids([self.hp, self.hp], ['iw_overlap_critic'])

    def test_matched_cases_and_temperature(self):
        hybrid = load_hybrids([self.hp], ['iw_overlap_critic'], [1.], [0])
        hybrid = hybrid[hybrid.bound_strategy == 'logits']
        baseline = load_baselines([self.bp], ['ATC_NE', 'COT', 'AC'], 'source')
        baseline = baseline[~((baseline.prediction_method == 'AC') & (baseline['shift'] == '2'))]
        groups = matched_comparison(hybrid, baseline, ['ATC_NE', 'COT', 'AC'])
        self.assertEqual([len(g) for g in groups], [1]*4)
        self.assertTrue(all(g['shift'].iloc[0] == '1' for g in groups))

    def test_inconsistent_target_truth_rejected(self):
        hybrid = load_hybrids([self.hp], ['iw_overlap_critic'], [1.], [0])
        hybrid = hybrid[hybrid.bound_strategy == 'logits']
        baseline = load_baselines([self.bp], ['AC'], 'source')
        baseline.loc[:, 'trg_accuracy'] = .1
        with self.assertRaisesRegex(ValueError, 'Target accuracies differ'):
            matched_comparison(hybrid, baseline, ['AC'])

    def test_missing_inputs_fail_before_writing_figures(self):
        args = ['--results', str(self.hp), '--plot_dir', str(self.root/'out')]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)
        data = hybrid_fixture()
        data[data.bound_strategy == 'logits'].to_csv(self.hp, index=False)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args + ['--figures', 'representations'])
        self.assertFalse((self.root/'out').exists())

    def test_complete_figures_include_baselines_and_all_pca_and_support_counts(self):
        data = hybrid_fixture()
        unsupported = ((data.prediction_method == 'iw_overlap_critic') & data.iw_threshold.eq(1)
                       & data.seed.eq(0) & data.bound_strategy.eq('PCA4') & data['shift'].eq(1))
        data.loc[unsupported, 'status'] = 'unsupported_critic_region'
        data.loc[unsupported, 'lower_bound'] = np.nan
        data.to_csv(self.hp, index=False)
        output = self.root/'figures'
        with contextlib.redirect_stdout(io.StringIO()):
            summary = main(['--results', str(self.hp), '--other_results_fname', str(self.bp),
                '--plot_dir', str(output), '--methods', 'iw_overlap_critic',
                '--iw_thresholds', '1', '--seeds', '0', '--da', 'no'])
        self.assertEqual(len(list(output.glob('*.png'))), 3)
        self.assertEqual(len(list(output.glob('*.pdf'))), 3)
        self.assertEqual(len(summary), 12)  # 4 comparison series + 2 full/logit + 6 PCA panels
        pca = summary[summary.panel == 'PCA4'].iloc[0]
        self.assertEqual(pca.n_unsupported_or_nonfinite, 1)
        self.assertEqual(pca.n_plotted, 0)
        points = pd.read_csv(next(output.glob('*_points.csv')))
        own = points[points.prediction_method == 'iw_overlap_critic']
        np.testing.assert_allclose(own.loc[own.plotted, 'prediction'], .4)
        self.assertEqual(set(points[points['plot'] == 'accuracy_vs_baselines'].prediction_method),
                         {'iw_overlap_critic', 'ATC_NE', 'COT', 'AC'})
        self.assertNotIn('dis2_reference', set(points.prediction_method))

    def test_allow_missing_marks_panels_without_fabricating_points(self):
        data = hybrid_fixture()
        data[data.bound_strategy == 'logits'].to_csv(self.hp, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = main(['--results', str(self.hp), '--plot_dir', str(self.root/'partial'),
                '--methods', 'iw_residual_dis2', '--iw_thresholds', '2', '--seeds', '1',
                '--figures', 'representations', '--allow_missing', '--da', 'yes'])
        self.assertEqual(summary.n_plotted.sum(), 1)
        self.assertEqual((summary.n_matched == 0).sum(), 7)


if __name__ == '__main__':
    unittest.main()
