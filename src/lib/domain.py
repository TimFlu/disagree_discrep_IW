"""Domain classification and density ratios; task labels are never used."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def density_ratio(probability, source_prior=0.5, target_prior=0.5, eps=1e-7):
    """Posterior odds corrected for the discriminator's effective sampling priors."""
    if not (0 < source_prior < 1 and 0 < target_prior < 1):
        raise ValueError('Domain priors must be in (0, 1).')
    if not math.isclose(source_prior + target_prior, 1.0):
        raise ValueError('Domain priors must sum to one.')
    if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Expected finite probabilities in [0, 1].')
    p = probability.to(torch.float64).clamp(eps, 1 - eps)
    return (source_prior / target_prior) * p / (1 - p)


class DomainRatio(nn.Module):
    """Linear discriminator, standardization and independent temperature calibration.

    Equal per-domain loss means enforce effective priors of 1/2 even when
    the two datasets have unequal sizes. Ratios use log odds to avoid sigmoid
    saturation. Numerical log-odds clipping is recorded by the evaluator.
    """
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('temperature', mean.new_tensor(1.0))
        self.head = nn.Linear(mean.numel(), 1).to(mean.device)

    def forward(self, x):
        return self.head((x - self.mean) / self.scale).squeeze(-1)

    @torch.no_grad()
    def log_odds(self, x):
        return self(x).to(torch.float64) / self.temperature

    @torch.no_grad()
    def ratios(self, x):
        return self.log_odds(x).clamp(-60, 60).exp()


def fit_domain(source_train, target_train, source_cal, target_cal,
               epochs=100, batch_size=256, lr=0.01):
    if min(map(len, (source_train, target_train, source_cal, target_cal))) < 2:
        raise ValueError('Need at least two samples per domain fitting/calibration split.')
    combined = torch.cat((source_train, target_train))
    model = DomainRatio(combined.mean(0), combined.std(0, unbiased=False).clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad()
        # One full objective per epoch; chunks limit memory, no domain is truncated.
        for features, domain in ((source_train, 0.), (target_train, 1.)):
            for x in features.split(batch_size):
                z = model(x)
                loss = F.binary_cross_entropy_with_logits(z, torch.full_like(z, domain), reduction='sum')
                (loss / (2 * len(features))).backward()
        optimizer.step()
    with torch.no_grad():
        zs, zt = model(source_cal).detach(), model(target_cal).detach()
    log_temperature = nn.Parameter(zs.new_zeros(()))
    calibration = torch.optim.LBFGS([log_temperature], max_iter=40, line_search_fn='strong_wolfe')

    def closure():
        calibration.zero_grad()
        temperature = log_temperature.clamp(-4, 4).exp()
        loss = (F.softplus(zs / temperature).mean() + F.softplus(-zt / temperature).mean()) / 2
        loss.backward()
        return loss

    calibration.step(closure)
    model.temperature.copy_(log_temperature.detach().clamp(-4, 4).exp())
    model.eval()
    with torch.no_grad():
        ps, pt = (zs / model.temperature).sigmoid(), (zt / model.temperature).sigmoid()
        diagnostics = {
            'domain_temperature': model.temperature.item(),
            'domain_source_prior': 0.5, 'domain_target_prior': 0.5,
            'domain_cal_balanced_accuracy': ((ps < .5).float().mean() + (pt >= .5).float().mean()).item() / 2,
            'domain_cal_brier': (ps.square().mean() + (1 - pt).square().mean()).item() / 2,
        }
    return model, diagnostics
