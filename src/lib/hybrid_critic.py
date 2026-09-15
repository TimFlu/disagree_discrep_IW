"""Independent critic fitting/selection for both hybrid methods.

Source weights are normalized to sum to one within the training region.
Target loss is the conditional mean on B. No task labels enter this module.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .critic import get_critic_source_loss, get_critic_target_margin


def _target_loss(logits, predictions, loss_type):
    if loss_type not in ('disagreement', 'DBAT', 'negative_xent'):
        raise ValueError('Unknown critic loss.')
    if loss_type == 'negative_xent':
        return -F.cross_entropy(logits, predictions, reduction='none')
    return F.softplus(get_critic_target_margin(logits, predictions, loss_type))


@torch.no_grad()
def _disagreements(weights, biases, features, predictions, sample_weights, batch_size):
    total = weights.new_zeros(len(weights))
    denominator = sample_weights.sum()
    for start in range(0, len(features), batch_size):
        x, p = features[start:start + batch_size], predictions[start:start + batch_size]
        logits = torch.einsum('nd,rdk->rnk', x, weights) + biases
        total += ((logits.argmax(-1) != p) * sample_weights[start:start + batch_size]).sum(1)
    return total / denominator


def train_hybrid_critic(source, target, source_select, target_select,
                        epochs=50, repeats=30, source_strength=1.,
                        batch_size=256, loss_type='disagreement'):
    """Each tuple is (features, frozen classifier logits, positive sample weights).

    Every sample contributes once per epoch to its domain mean. Select one
    repeat AND epoch using held-out d_target - source_strength*d_source.
    """
    if epochs < 1 or repeats < 1 or batch_size < 1 or source_strength <= 0:
        raise ValueError('Training counts and source strength must be positive.')
    for x, p, w in (source, target, source_select, target_select):
        if not len(x) or len(x) != len(p) or len(x) != len(w) or w.sum() <= 0:
            raise ValueError('Critic regions must be nonempty with positive total weights.')
    dim = source[0].shape[1]
    # Frozen logits retain classes absent from a selected region.
    classes = source[1].shape[1]
    data = [(x, logits.argmax(1), w.to(x.dtype)) for x, logits, w in (source, target, source_select, target_select)]
    source, target, source_select, target_select = data
    weights = nn.Parameter(source[0].new_empty(repeats, dim, classes).uniform_(-1 / math.sqrt(dim), 1 / math.sqrt(dim)))
    biases = nn.Parameter(source[0].new_zeros(repeats, 1, classes))
    optimizer = torch.optim.AdamW([weights, biases], lr=3e-3, weight_decay=5e-4)
    best_score, best_model, history = -float('inf'), None, []
    for epoch in range(epochs):
        optimizer.zero_grad()
        for is_source, (x, p, w) in ((True, source), (False, target)):
            denom = w.sum()
            for start in range(0, len(x), batch_size):
                xb, pb, wb = x[start:start + batch_size], p[start:start + batch_size], w[start:start + batch_size]
                logits = torch.einsum('nd,rdk->rnk', xb, weights) + biases
                flat, labels = logits.flatten(0, 1), pb.repeat(repeats)
                losses = get_critic_source_loss(flat, labels) if is_source else _target_loss(flat, labels, loss_type)
                domain_loss = (losses.reshape(repeats, -1) * wb).sum() / denom
                (domain_loss * (source_strength if is_source else 1.) / 2).backward()
        optimizer.step()
        ds = _disagreements(weights, biases, *source_select, batch_size)
        dt = _disagreements(weights, biases, *target_select, batch_size)
        scores = dt - source_strength * ds
        index = int(scores.argmax())
        score = float(scores[index])
        history.append({'epoch': epoch, 'best_score': score})
        if score > best_score:
            best_score = score
            best_model = nn.Linear(dim, classes, device=weights.device)
            with torch.no_grad():
                best_model.weight.copy_(weights[index].T)
                best_model.bias.copy_(biases[index, 0])
            selected = {'critic_epoch': epoch, 'critic_repeat': index, 'critic_selection_score': score,
                        'critic_selection_source_disagreement': float(ds[index]),
                        'critic_selection_target_disagreement': float(dt[index])}
    if best_model is None:
        raise RuntimeError('Critic training produced no finite selection score.')
    best_model.eval()
    return best_model, {**selected, 'critic_history': history}
