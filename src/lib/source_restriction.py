"""Source restriction changes the source constraints; target is never restricted."""
import numpy as np
import torch

from .domain import fit_domain
from .hybrid_critic import train_hybrid_critic
from .hybrid_validation import predict_disagreement

METHODS = ('source_pruning', 'source_weighted_critic')
REFERENCE = 'source_restriction_dis2_reference'


def restriction_splits(n, seed, fractions=(.4, .1)):
    """Two outer blocks: fit/development (with internal validation), final eval."""
    if len(fractions) != 2 or not np.isfinite(fractions).all() or min(fractions) <= 0 or sum(fractions) >= 1:
        raise ValueError('fit and validation fractions must be positive and sum to less than one.')
    sizes = [int(n * f) for f in fractions]
    if min(*sizes, n - sum(sizes)) < 2:
        raise ValueError('Need at least two samples in each fit/validation/evaluation split.')
    return dict(zip(('fit', 'validation', 'eval'),
                    np.split(np.random.default_rng(seed).permutation(n), np.cumsum(sizes))))


def source_mask(probabilities, threshold):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected finite source target-probabilities in [0,1].')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Source threshold must be in [0,1].')
    return p >= threshold


def dis2_components(source_errors, source_disagreement, target_disagreement, delta=.01):
    """Unweighted DIS2 final formula; caller passes either pruned or full source."""
    if not 0 < delta < 1:
        raise ValueError('delta must be between 0 and 1.')
    arrays = [np.asarray(a, dtype=float) for a in (source_errors, source_disagreement, target_disagreement)]
    if any(a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any()
           for a in arrays) or arrays[0].shape != arrays[1].shape:
        raise ValueError('Expected aligned nonempty source arrays and nonempty target disagreements in [0,1].')
    es, ds, dt = (float(a.mean()) for a in arrays)
    ns, nt = len(arrays[0]), len(arrays[2])
    eps = float(np.sqrt((ns + 4*nt) * np.log(1/delta) / (2*ns*nt)))
    raw = es + dt - ds
    return dict(source_error=es, source_disagreement=ds, target_disagreement=dt,
                max_ts_agree_diff=dt-ds, h_val_acc=1-es,
                n_val_source=ns, n_val_target=nt, epsilon=eps, delta=delta,
                error_estimate_raw=raw, accuracy_estimate_raw=1-raw,
                error_upper_bound_raw=raw+eps, accuracy_lower_bound_raw=1-raw-eps,
                lower_bound=float(np.clip(1-raw-eps, 0, 1)))


def evaluate_source_restriction(source_features, source_logits, source_labels,
        target_features, target_logits, *, methods=METHODS, thresholds=(.5, .7, .9),
        include_reference=False, seed=0, fractions=(.4, .1), domain_epochs=100,
        epochs=50, repeats=30, batch_size=256, loss_type='disagreement', delta=.01,
        min_source_samples=2):
    """No target task labels: they belong exclusively to downstream reporting.

    The discriminator fits on fit, calibrates on validation, then is frozen.
    Critics reuse fit and validation; the entire eval split is held out.
    Soft source probabilities weight BOTH critic fitting and selection, but
    never the final reported error/disagreement means or correction.
    """
    if not methods or any(m not in METHODS for m in methods):
        raise ValueError('Choose source_pruning and/or source_weighted_critic.')
    if len(set(methods)) != len(methods) or len(set(thresholds)) != len(thresholds):
        raise ValueError('Methods and thresholds must be unique.')
    if not 0 < delta < 1 or min_source_samples < 1:
        raise ValueError('Require 0 < delta < 1 and min_source_samples >= 1.')
    if min(domain_epochs, epochs, repeats, batch_size) < 1:
        raise ValueError('Training counts must be positive.')
    if 'source_pruning' in methods and not len(thresholds):
        raise ValueError('Pruning requires at least one source threshold.')
    for threshold in thresholds:
        source_mask(np.ones(1), threshold)
    for x, logits in ((source_features, source_logits), (target_features, target_logits)):
        if x.ndim != 2 or logits.ndim != 2 or len(x) != len(logits) or logits.shape[1] < 2:
            raise ValueError('Features and logits must be aligned matrices with at least two classes.')
        if not torch.isfinite(x).all() or not torch.isfinite(logits).all():
            raise ValueError('Features and logits must be finite.')
    if source_features.shape[1] != target_features.shape[1] or source_logits.shape[1] != target_logits.shape[1]:
        raise ValueError('Source and target dimensions must match.')
    if source_labels.ndim != 1 or len(source_labels) != len(source_features):
        raise ValueError('Source labels must align with source features.')
    s = restriction_splits(len(source_features), seed, fractions)
    t = restriction_splits(len(target_features), seed+1, fractions)
    torch.manual_seed(seed)
    domain, diagnostics = fit_domain(source_features[s['fit']], target_features[t['fit']],
                                    source_features[s['validation']], target_features[t['validation']],
                                    epochs=domain_epochs, batch_size=batch_size)
    with torch.no_grad():
        # Bounded posterior weights, NOT odds or importance weights.
        scores = domain.log_odds(source_features).sigmoid().cpu().numpy()
    source_mask(scores, 0.)  # Validate scores without changing them.
    jobs = [(m, tau) for m in methods for tau in (thresholds if m == 'source_pruning' else [np.nan])]
    if include_reference:
        jobs.append((REFERENCE, np.nan))
    se, te = s['eval'], t['eval']
    source_errors = (source_logits.argmax(1) != source_labels).cpu().numpy()
    rows, models = [], {}
    for method, threshold in jobs:
        mask = source_mask(scores, threshold) if method == 'source_pruning' else np.ones(len(scores), dtype=bool)
        selected = {name: idx[mask[idx]] for name, idx in s.items()}
        model_key = f'{method}_tau{threshold:g}' if np.isfinite(threshold) else method
        row = dict(prediction_method=method, source_threshold=threshold, seed=seed,
            n_source=len(source_features), n_target=len(target_features),
            n_source_eval_full=len(se), n_target_eval_full=len(te),
            n_val_source=len(selected['eval']), n_val_target=len(te),
            source_retained_fraction=len(selected['eval'])/len(se),
            n_source_fit_used=len(selected['fit']), n_source_validation_used=len(selected['validation']),
            n_target_fit_used=len(t['fit']), n_target_validation_used=len(t['validation']),
            source_weighting='target_probability_normalized' if method == 'source_weighted_critic' else 'uniform',
            final_source_weighting='uniform', target_weighting='uniform',
            split_protocol='shared_fit_validation_heldout_eval',
            estimate_kind='dis2_with_correction', guarantee_status='conditional_on_dis2_critic_dominance',
            confidence_bound_available=False, status='ok', delta=delta,
            model_key=model_key, **diagnostics)
        train_weights = scores[selected['fit']] if method == 'source_weighted_critic' else np.ones(len(selected['fit']))
        row.update(source_fit_weight_sum=float(train_weights.sum()),
                   source_fit_effective_n=float(train_weights.sum()**2/np.square(train_weights).sum())
                   if np.square(train_weights).sum() else 0.,
                   source_score_eval_mean=float(scores[se].mean()),
                   source_score_eval_p10=float(np.quantile(scores[se], .1)),
                   source_score_eval_p90=float(np.quantile(scores[se], .9)))
        for name in ('eval', 'fit', 'validation'):
            if len(selected[name]) < min_source_samples:
                row['status'] = f'unsupported_source_{name}'
                break
        groups = []
        if row['status'] == 'ok':
            for name in ('fit', 'validation'):
                si, ti = selected[name], t[name]
                weight = scores[si] if method == 'source_weighted_critic' else np.ones(len(si))
                weight = torch.as_tensor(weight, dtype=source_features.dtype, device=source_features.device)
                if not torch.isfinite(weight).all() or weight.sum() <= 0:
                    row['status'] = 'unsupported_source_weights'
                    break
                groups.extend([(source_features[si], source_logits[si], weight),
                               (target_features[ti], target_logits[ti], target_features.new_ones(len(ti)))])
        if row['status'] != 'ok':
            row.update({key: np.nan for key in ['source_error','source_disagreement','target_disagreement',
                'max_ts_agree_diff','h_val_acc','epsilon','error_estimate_raw','accuracy_estimate_raw',
                'error_upper_bound_raw','accuracy_lower_bound_raw','lower_bound']})
        else:
            torch.manual_seed(seed+2)  # Comparable initialization for each variant/reference.
            critic, stats = train_hybrid_critic(*groups, epochs=epochs, repeats=repeats,
                                              batch_size=batch_size, loss_type=loss_type)
            si = selected['eval']
            ds = predict_disagreement(critic, source_features[si], source_logits[si], batch_size)
            dt = predict_disagreement(critic, target_features[te], target_logits[te], batch_size)
            row.update(dis2_components(source_errors[si], ds, dt, delta), **stats,
                       confidence_bound_available=True)
            models[model_key] = {key: value.detach().cpu() for key, value in critic.state_dict().items()}
        rows.append(row)
    artifacts = dict(source_scores=scores, source_masks={f'tau{tau:g}':source_mask(scores,tau)
                      for tau in thresholds} if 'source_pruning' in methods else {},
                     domain_state={k:v.detach().cpu() for k,v in domain.state_dict().items()}, critics=models)
    return rows, {'source':s, 'target':t}, artifacts
