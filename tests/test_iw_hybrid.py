import inspect
import numpy as np
import pytest
import torch
from src.lib.domain import density_ratio, fit_domain
from src.lib.overlap import iw_contribution, select_region
from src.lib.hybrid_validation import aggregate, split_indices, evaluate_hybrids, evaluate_dis2_reference, METHODS
from src.lib.hybrid_critic import train_hybrid_critic


torch.set_num_threads(1)


def test_oracle_mass_shift_recovers_sixty_percent():
    # Same labeling rule; error only on A: source mass .1, target mass .6.
    errors = np.r_[np.ones(10), np.zeros(90)]
    weights = np.r_[np.full(10, 6.), np.full(90, 4 / 9)]
    for method in METHODS:
        row = aggregate(method, errors, weights, np.ones(100, bool), np.ones(100, bool))
        assert row['error_estimate_raw'] == pytest.approx(.6)
        assert row['lower_bound'] == pytest.approx(.4)
        assert row['estimate_kind'] == 'plugin'
        assert not row['confidence_bound_available']


def test_full_denominator_and_residual_mass():
    errors = np.array([1., 0., 1., 0.])
    weights = np.array([.4, .4, 2., 2.])
    sa, ta = np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 0], bool)
    ds, dt = np.array([0, 0, 1, 0]), np.array([0, 1, 1, 0])
    a = aggregate(METHODS[0], errors, weights, sa, ta, ds, dt)
    b = aggregate(METHODS[1], errors, weights, sa, ta, ds, dt)
    assert a['iw_error_contribution'] == pytest.approx(.1)  # .4 / 4, not .4 / 2
    assert a['residual_error_contribution'] == pytest.approx(.5)  # .75 * (2/3)
    assert b['error_estimate_raw'] == pytest.approx(.1 + .75 * (.5 + 2 / 3 - .25))


def test_same_critic_difference_identity():
    e, ds, dt = np.array([0, 1, 0, 0]), np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    sa, ta = np.array([1, 0, 0, 0], bool), np.array([1, 1, 0, 0], bool)
    a = aggregate(METHODS[0], e, np.ones(4), sa, ta, ds, dt)
    b = aggregate(METHODS[1], e, np.ones(4), sa, ta, ds, dt)
    assert b['error_estimate_raw'] - a['error_estimate_raw'] == pytest.approx(.5 * (.25 - .5))


def test_no_iw_region_reduces_to_dis2_expression():
    e, ds, dt = [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1]
    row = aggregate(METHODS[1], e, np.ones(4), np.zeros(4, bool), np.zeros(3, bool), ds, dt)
    assert row['error_estimate_raw'] == pytest.approx(.25 + 1 - .5)


def test_empty_source_nonoverlap_uses_full_source():
    row = aggregate(METHODS[1], [1, 0], [.5, .5], [True, True], [False, False],
                    [0, 0], [1, 0])
    assert row['status'] == 'ok'
    assert row['residual_source_region'] == 'full'
    assert row['residual_source_error'] == .5
    assert row['error_estimate_raw'] == pytest.approx(.25 + .5 + .5)


def test_missing_iw_source_is_unsupported():
    row = aggregate(METHODS[0], [0, 0], [2, 2], [False, False], [True, True])
    assert row['status'] == 'unsupported_iw_source_region'


def test_raw_out_of_range_is_preserved():
    row = aggregate(METHODS[0], [1, 1], [3, 3], [True, True], [True, True])
    assert row['error_estimate_raw'] == 3
    assert row['lower_bound'] == 0
    assert row['output_clipped']


def test_density_ratio_prior_correction_and_threshold():
    # p_T/p_S = 3, with sampling priors pi_S=.8, pi_T=.2 -> d=3/7.
    ratio = density_ratio(torch.tensor([3 / 7], dtype=torch.float64), .8, .2)
    assert ratio.item() == pytest.approx(3)
    assert select_region([.5, 2, 10], 2).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        select_region([np.inf], 2)
    with pytest.raises(ValueError):
        iw_contribution([1, 0], [1], [True, False])


def test_nested_splits_are_reproducible_and_evaluation_is_independent():
    for n, fractions in ((201, (.4, .1, .2)), (300, (.3, .2, .1))):
        split = split_indices(n, 7, fractions)
        atomic = [split[k] for k in ('domain_train', 'domain_cal', 'critic_select', 'eval')]
        assert len(np.concatenate(atomic)) == len(np.unique(np.concatenate(atomic))) == n
        np.testing.assert_array_equal(split['critic_train'], np.concatenate(atomic[:2]))
        for key, values in split.items():
            np.testing.assert_array_equal(values, split_indices(n, 7, fractions)[key])
        assert not set(split['critic_train']) & set(split['critic_select'])
        assert not set(split['critic_train']) & set(split['eval'])
    split = split_indices(200, 0)
    assert {k: len(v) for k, v in split.items()} == dict(
        domain_train=80, domain_cal=20, critic_train=100, critic_select=40, eval=60)
    for fractions in ((.2, .1, .3, .15), (.4, float('nan'), .2), (.4, .1, .5), (.4, 0, .2)):
        with pytest.raises(ValueError):
            split_indices(200, 0, fractions)
    with pytest.raises(ValueError):
        split_indices(10, 0)


def critic_data():
    torch.manual_seed(9)
    def group(n):
        x = torch.randn(n, 3)
        return x, torch.stack((x[:, 0], -x[:, 0]), 1), torch.rand(n) + .1
    return [group(n) for n in (23, 41, 19, 31)]


def test_selected_checkpoint_matches_reported_score():
    groups = critic_data()
    model, stats = train_hybrid_critic(*groups, repeats=3, epochs=8, batch_size=7, source_strength=.7)
    with torch.no_grad():
        d = [float(((model(x).argmax(1) != logits.argmax(1)) * w).sum() / w.sum())
             for x, logits, w in groups[2:]]
    assert d[1] - .7 * d[0] == pytest.approx(stats['critic_selection_score'], abs=1e-6)
    assert stats['critic_selection_score'] == max(h['best_score'] for h in stats['critic_history'])


def test_batching_does_not_truncate_unequal_domains():
    groups = critic_data()
    torch.manual_seed(21)
    a, _ = train_hybrid_critic(*groups, repeats=2, epochs=3, batch_size=5)
    torch.manual_seed(21)
    b, _ = train_hybrid_critic(*groups, repeats=2, epochs=3, batch_size=100)
    torch.testing.assert_close(a.weight, b.weight, atol=1e-6, rtol=1e-5)


def synthetic_data():
    torch.manual_seed(18)
    sf, tf = torch.randn(600, 3), torch.randn(800, 3) + torch.tensor([1.5, 0., 0.])
    sl, tl = torch.stack((sf[:, 1], -sf[:, 1]), 1), torch.stack((tf[:, 1], -tf[:, 1]), 1)
    sy = ((sf[:, 1] + .5 * sf[:, 0]) < 0).long()
    return sf, sl, sy, tf, tl


def test_domain_fit_and_hybrid_integration():
    sf, sl, sy, tf, tl = synthetic_data()
    rows, splits = evaluate_hybrids(sf, sl, sy, tf, tl, thresholds=[1., 2.],
                              domain_epochs=100, epochs=3, repeats=2, batch_size=100)
    assert len(rows) == 4
    assert all(r['status'] == 'ok' for r in rows)
    assert all(np.isfinite(r['lower_bound']) for r in rows)
    assert all('critic_epoch' in r for r in rows)
    for row in rows:
        if row['prediction_method'] == METHODS[1]:
            for split in ('critic_train', 'critic_select'):
                assert row[f'n_source_{split}_region'] == len(splits['source'][split])
            assert row['residual_source_error'] == pytest.approx(1 - row['h_val_acc'])
    assert 'target_labels' not in inspect.signature(evaluate_hybrids).parameters
    ref = evaluate_dis2_reference(sf, sl, sy, tf, tl, epochs=3, repeats=2)
    assert ref['epsilon'] > 0
    assert ref['n_val_source'] == rows[0]['n_val_source']


def test_identical_domains_with_unequal_counts_have_unit_ratio():
    # Equal empirical domain distributions despite different counts.
    torch.manual_seed(4)
    x = torch.randn(60, 2)
    model, diagnostics = fit_domain(x, x.repeat(3, 1), x, x.repeat(2, 1), epochs=80)
    assert diagnostics['domain_source_prior'] == .5
    assert float((model.ratios(x) - 1).abs().mean()) < .1


def test_cli_files_plot_and_target_label_isolation(tmp_path):
    import pandas as pd
    from src.eval.iw_hybrid import main
    from src.plot.compare_iw_hybrids import main as plot_main, load_results
    sf, sl, sy, tf, tl = synthetic_data()
    features = tmp_path / 'features'
    shift = features / 'cifar10' / 'ERM-aug-imagenet_1_100.0'
    shift.mkdir(parents=True)
    np.save(shift / 'model_feats.npy', {'source': sf.numpy(), 'target': tf.numpy()})
    labels = {'source': sy.numpy(), 'target': (tf[:, 1] < 0).long().numpy()}
    np.save(shift / 'ytrue.npy', labels)
    head = torch.nn.Linear(3, 2)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0., 1., 0.], [0., -1., 0.]]))
        head.bias.zero_()
    torch.save(head.state_dict(), shift / 'linear.pth')
    shared = ['--feats_dir', str(features), '--datasets', 'cifar10', '--shifts', '1',
              '--train_methods', 'ERM-aug-imagenet', '--bound_strategies', 'features',
              '--domain_epochs', '100', '--epochs', '3', '--critic_repeats', '2',
              '--iw_thresholds', '1', '2', '--include_dis2_reference', '--device', 'cpu']
    first = main(shared + ['--results_dir', str(tmp_path / 'run1')])
    a = pd.read_pickle(first)
    assert len(a) == 5 and (a.status == 'ok').all()
    assert (a.schema_version == 3).all()
    assert (a.loc[a.prediction_method == METHODS[1], 'residual_source_region'] == 'full').all()
    assert first.with_suffix('.csv').exists()
    assert len(list(first.parent.glob('splits_*.npz'))) == 1
    labels['target'] = 1 - labels['target']
    np.save(shift / 'ytrue.npy', labels)
    second = main(shared + ['--results_dir', str(tmp_path / 'run2')])
    b = pd.read_pickle(second)
    np.testing.assert_array_equal(a.lower_bound.to_numpy(), b.lower_bound.to_numpy())
    np.testing.assert_array_equal(a.error_estimate_raw.to_numpy(), b.error_estimate_raw.to_numpy())
    assert not np.array_equal(a.trg_accuracy, b.trg_accuracy)
    plots = tmp_path / 'plots'
    plot_main(['--results', str(first), '--plot_dir', str(plots), '--bound_strategy', 'features'])
    assert (plots / 'compare_iw_hybrids_features.png').exists()
    assert (plots / 'compare_iw_hybrids_features_summary.csv').exists()
    # Loader must preserve saved results, even when historical formula columns differ.
    np.testing.assert_array_equal(load_results([first]).lower_bound, a.lower_bound)


def test_residual_critic_keeps_full_source_when_source_b_is_empty(monkeypatch):
    import src.lib.hybrid_validation as validation
    sf, sl, sy, tf, tl = synthetic_data()

    class FixedRatios:
        def ratios(self, x):
            return x.new_full((len(x),), .5 if x is sf else 2.)

        def log_odds(self, x):
            return x.new_zeros(len(x))

    monkeypatch.setattr(validation, 'fit_domain', lambda *a, **k: (FixedRatios(), {}))
    original_train = validation.train_hybrid_critic
    captured = []

    def capture(*groups, **kwargs):
        captured.extend(groups)
        return original_train(*groups, **kwargs)

    monkeypatch.setattr(validation, 'train_hybrid_critic', capture)
    rows, splits = evaluate_hybrids(sf, sl, sy, tf, tl, methods=[METHODS[1]],
                                   thresholds=[1.], epochs=2, repeats=1)
    row = rows[0]
    assert row['status'] == 'ok'
    assert row['n_source_b'] == 0
    assert row['target_residual_mass'] == 1
    for group, split in zip(captured[::2], ('critic_train', 'critic_select')):
        x, logits, weights = group
        torch.testing.assert_close(x, sf[splits['source'][split]])
        torch.testing.assert_close(logits, sl[splits['source'][split]])
        assert torch.all(weights == 1)
    assert row['residual_source_error'] == pytest.approx(1 - row['h_val_acc'])
