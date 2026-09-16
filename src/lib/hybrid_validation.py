"""Evaluation of plug-in hybrid expressions (not confidence bounds)."""
import numpy as np
import torch
from .domain import fit_domain
from .overlap import select_region, iw_contribution, weight_diagnostics
from .hybrid_critic import train_hybrid_critic

METHODS = ('iw_overlap_critic', 'iw_residual_dis2')
SPLITS = ('domain_train', 'domain_cal', 'critic_train', 'critic_select', 'eval')


def split_indices(n, seed, fractions=(.2, .1, .3, .15)):
    if len(fractions) != 4 or min(fractions) <= 0 or sum(fractions) >= 1:
        raise ValueError('Four positive fitting fractions must sum to less than one.')
    sizes = [int(n * f) for f in fractions]
    sizes.append(n - sum(sizes))
    if min(sizes) < 2:
        raise ValueError('Each split needs at least two samples; increase data or adjust fractions.')
    indices = np.random.default_rng(seed).permutation(n)
    return dict(zip(SPLITS, np.split(indices, np.cumsum(sizes)[:-1])))


def aggregate(method, source_errors, weights, source_a, target_a,
              source_disagreement=None, target_disagreement=None):
    """Pure empirical algebra. Inputs use the FULL final-evaluation arrays."""
    if method not in METHODS:
        raise ValueError('Unknown hybrid method.')
    source_errors, weights = np.asarray(source_errors), np.asarray(weights)
    source_a, target_a = np.asarray(source_a, dtype=bool), np.asarray(target_a, dtype=bool)
    if not len(target_a):
        raise ValueError('Target evaluation set must be nonempty.')
    iw = iw_contribution(source_errors, weights, source_a)
    mass = float((~target_a).mean())
    out = {'iw_error_contribution': iw, 'target_residual_mass': mass,
           'n_source_a': int(source_a.sum()), 'n_source_b': int((~source_a).sum()),
           'n_target_a': int(target_a.sum()), 'n_target_b': int((~target_a).sum()),
           'status': 'ok', 'estimate_kind': 'plugin', 'confidence_bound_available': False,
           'residual_source_region': 'full' if method == METHODS[1] else 'not_applicable',
           'epsilon': np.nan, **weight_diagnostics(weights, source_a)}
    if target_a.any() and not source_a.any():
        out['status'] = 'unsupported_iw_source_region'
    residual = 0.
    if out['status'] == 'ok' and mass:
        if target_disagreement is None or np.asarray(target_disagreement).shape != target_a.shape:
            raise ValueError('Aligned target disagreements required for nonempty B.')
        dt = float(np.asarray(target_disagreement)[~target_a].mean())
        out['residual_target_disagreement'] = dt
        residual = mass * dt
        if method == 'iw_residual_dis2':
            if source_disagreement is None or np.asarray(source_disagreement).shape != source_a.shape:
                raise ValueError('Aligned source disagreements required for residual DIS2.')
            # Residual DIS2 compares full S against conditional target B.
            es = float(source_errors.mean())
            ds = float(np.asarray(source_disagreement).mean())
            out.update(residual_source_error=es, residual_source_disagreement=ds,
                       residual_discrepancy=dt - ds)
            residual = mass * (es + dt - ds)
    raw = iw + residual if out['status'] == 'ok' else np.nan
    out.update(residual_error_contribution=residual if out['status'] == 'ok' else np.nan,
               error_estimate_raw=raw, accuracy_estimate_raw=1 - raw,
               lower_bound=float(np.clip(1 - raw, 0, 1)),
               output_clipped=bool(np.isfinite(raw) and not 0 <= raw <= 1))
    return out


@torch.no_grad()
def predict_disagreement(critic, features, logits, batch_size):
    return torch.cat([(critic(x).argmax(1) != h.argmax(1)).cpu()
                      for x, h in zip(features.split(batch_size), logits.split(batch_size))]).numpy()


def evaluate_hybrids(source_features, source_logits, source_labels, target_features, target_logits,
                     methods=METHODS, thresholds=(1., 2., 5.), seed=0,
                     fractions=(.2, .1, .3, .15), domain_epochs=100, epochs=50,
                     repeats=30, batch_size=256, source_strength=1., loss_type='disagreement'):
    """Target task labels deliberately absent; caller may attach benchmark truth.

    The selection region is fitted in the representation supplied by the caller.
    Returns rows and split indices for reproducible, aligned benchmark reporting.
    """
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError('Choose at least one supported method.')
    if domain_epochs < 1 or epochs < 1 or repeats < 1 or batch_size < 1:
        raise ValueError('Training counts must be positive.')
    if not np.isfinite(source_strength) or source_strength <= 0:
        raise ValueError('Source strength must be positive and finite.')
    if not thresholds:
        raise ValueError('At least one IW threshold is required.')
    for threshold in thresholds:
        select_region(np.ones(1), threshold)
    for features, logits in ((source_features, source_logits), (target_features, target_logits)):
        if features.ndim != 2 or logits.ndim != 2 or len(features) != len(logits) or logits.shape[1] < 2:
            raise ValueError('Aligned features and multiclass logits are required.')
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise ValueError('Nonfinite features or logits.')
    if source_features.shape[1] != target_features.shape[1] or source_logits.shape[1] != target_logits.shape[1]:
        raise ValueError('Domain dimensions must match.')
    if len(source_labels) != len(source_features):
        raise ValueError('Source labels must align with features.')
    torch.manual_seed(seed)
    s = split_indices(len(source_features), seed, fractions)
    t = split_indices(len(target_features), seed + 1, fractions)
    ratio_model, domain_stats = fit_domain(source_features[s['domain_train']], target_features[t['domain_train']],
                                          source_features[s['domain_cal']], target_features[t['domain_cal']],
                                          epochs=domain_epochs, batch_size=batch_size)
    sw = ratio_model.ratios(source_features).cpu().numpy()
    tw = ratio_model.ratios(target_features).cpu().numpy()
    clipped = int((ratio_model.log_odds(source_features).abs() > 60).sum()
                  + (ratio_model.log_odds(target_features).abs() > 60).sum())
    se, te = s['eval'], t['eval']
    errors = (source_logits[se].argmax(1) != source_labels[se]).cpu().numpy()
    rows = []
    for threshold in thresholds:
        sa, ta = select_region(sw, threshold), select_region(tw, threshold)
        for method in methods:
            common = {'prediction_method': method, 'iw_threshold': threshold, 'seed': seed,
                      'residual_source_region': 'full' if method == METHODS[1] else 'not_applicable',
                      'n_source': len(source_features), 'n_target': len(target_features),
                      'n_val_source': len(se), 'n_val_target': len(te),
                      'h_val_acc': float(1 - errors.mean()), 'domain_log_odds_clipped_count': clipped,
                      'source_strength': source_strength if method == METHODS[0] else 1.,
                      **domain_stats}
            ds, dt, training_stats, status = None, None, {}, None
            if (~ta[te]).any() and not (ta[te].any() and not sa[se].any()):
                source_mask = sa if method == METHODS[0] else np.ones_like(sa, dtype=bool)
                groups = []
                for split in ('critic_train', 'critic_select'):
                    si, ti = s[split][source_mask[s[split]]], t[split][~ta[t[split]]]
                    common[f'n_source_{split}_region'] = len(si)
                    common[f'n_target_{split}_region'] = len(ti)
                    weight = sw[si] if method == METHODS[0] else np.ones(len(si))
                    groups.extend([(source_features[si], source_logits[si], torch.as_tensor(weight, device=source_features.device)),
                                   (target_features[ti], target_logits[ti], target_features.new_ones(len(ti)))])
                if any(len(x) == 0 for x, _, _ in groups):
                    status = 'unsupported_critic_region'
                else:
                    # Identical starting RNG across thresholds/methods; regions differ.
                    torch.manual_seed(seed + 2)
                    critic, training_stats = train_hybrid_critic(*groups, epochs=epochs, repeats=repeats,
                        source_strength=common['source_strength'], batch_size=batch_size, loss_type=loss_type)
                    ds = predict_disagreement(critic, source_features[se], source_logits[se], batch_size)
                    dt = predict_disagreement(critic, target_features[te], target_logits[te], batch_size)
            if status:
                row = {**common, **weight_diagnostics(sw[se], sa[se]), 'status': status,
                       'estimate_kind': 'plugin', 'confidence_bound_available': False, 'epsilon': np.nan,
                       'lower_bound': np.nan, 'error_estimate_raw': np.nan, 'accuracy_estimate_raw': np.nan,
                       'iw_error_contribution': iw_contribution(errors, sw[se], sa[se]),
                       'target_residual_mass': float((~ta[te]).mean()),
                       'n_source_a': int(sa[se].sum()), 'n_source_b': int((~sa[se]).sum()),
                       'n_target_a': int(ta[te].sum()), 'n_target_b': int((~ta[te]).sum())}
            else:
                row = {**common, **aggregate(method, errors, sw[se], sa[se], ta[te], ds, dt), **training_stats}
            rows.append(row)
    return rows, {'source': s, 'target': t}


def evaluate_dis2_reference(source_features, source_logits, source_labels, target_features, target_logits,
                            seed=0, fractions=(.2, .1, .3, .15), epochs=50, repeats=30,
                            batch_size=256, loss_type='disagreement', delta=.01):
    """Full-domain DIS2 with the same independent critic/evaluation splits.

    Separate from the untouched historical runner; enables fair split comparisons.
    The correction retains the original DIS2 assumptions.
    """
    if not 0 < delta < 1:
        raise ValueError('delta must be in (0, 1).')
    s, t = split_indices(len(source_features), seed, fractions), split_indices(len(target_features), seed + 1, fractions)
    groups = []
    for name in ('critic_train', 'critic_select'):
        groups.extend([(source_features[s[name]], source_logits[s[name]], source_features.new_ones(len(s[name]))),
                       (target_features[t[name]], target_logits[t[name]], target_features.new_ones(len(t[name])))])
    torch.manual_seed(seed + 2)
    critic, stats = train_hybrid_critic(*groups, epochs=epochs, repeats=repeats,
                                      batch_size=batch_size, loss_type=loss_type)
    se, te = s['eval'], t['eval']
    es = float((source_logits[se].argmax(1) != source_labels[se]).float().mean())
    ds = float(predict_disagreement(critic, source_features[se], source_logits[se], batch_size).mean())
    dt = float(predict_disagreement(critic, target_features[te], target_logits[te], batch_size).mean())
    ns, nt = len(se), len(te)
    epsilon = float(np.sqrt((ns + 4 * nt) * np.log(1 / delta) / (2 * ns * nt)))
    raw = es + dt - ds
    row = {'prediction_method': 'dis2_reference', 'seed': seed, 'iw_threshold': np.nan,
           'status': 'ok', 'estimate_kind': 'dis2_with_correction',
           'confidence_bound_available': True, 'epsilon': epsilon, 'delta': delta,
           'error_estimate_raw': raw, 'accuracy_estimate_raw': 1 - raw,
           'error_upper_bound_raw': raw + epsilon, 'lower_bound': float(np.clip(1 - raw - epsilon, 0, 1)),
           'h_val_acc': 1 - es, 'max_ts_agree_diff': dt - ds,
           'n_source': len(source_features), 'n_target': len(target_features),
           'n_val_source': ns, 'n_val_target': nt, **stats}
    return row
