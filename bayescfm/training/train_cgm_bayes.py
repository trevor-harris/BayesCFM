import math, torch
from ..losses.cfm import cgm_total_loss_ad

class _SGHMCState:
    def __init__(self, p): self.v = torch.zeros_like(p)

def _sghmc_update(param, grad, state, eta, temperature, friction, weight_decay):
    if weight_decay != 0.0:
        grad = grad.add(param, alpha=weight_decay)
    alpha = friction
    noise_std = math.sqrt(2.0 * alpha * eta * temperature)
    state.v.mul_(1.0 - alpha).add_(grad, alpha=-eta).add_(torch.randn_like(param) * noise_std)
    param.add_(state.v)

def train_cgm_bayes(
    model, loader, *, epochs=1, lr=2e-4, device='cuda', weight_decay=0.0,
    lambda_curl=0.0, lambda_mono=0.0,
    probes=1, probe_dist='rademacher', normalize_probes=True, orthogonalize=True, normalize_curl=True,
    penalty_train_flag=True, pool_factor=None, pool_to=None, enforce_stride=True,
    sgmcmc_enable=True, sgmcmc_alg='sghmc', sgmcmc_eta=2e-5, sgmcmc_temperature=1.0, sgmcmc_friction=0.05,
    sgmcmc_collect=True, sgmcmc_burnin_steps=100, sgmcmc_thin=20, sgmcmc_max_samples=20,
    grad_clip=0.0, return_posterior=True,
):
    model.to(device)
    if not sgmcmc_enable:
        raise ValueError("SG-MCMC disabled; use train_cgm for deterministic training.")
    params = [p for p in model.parameters() if p.requires_grad]
    states = [ _SGHMCState(p) for p in params ]
    ema = torch.optim.swa_utils.AveragedModel(model)

    posterior = []
    step = 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x1 = batch[0].to(device); y = batch[1].to(device) if len(batch) > 1 else None
            elif isinstance(batch, dict):
                x1 = batch.get('images').to(device); y = batch.get('labels', None); y = y.to(device) if y is not None else None
            else:
                x1, y = batch.to(device), None

            t = torch.rand(x1.size(0), device=device)
            loss, logs = cgm_total_loss_ad(
                model, x1, t, y,
                lambda_curl=lambda_curl, lambda_mono=lambda_mono,
                probes=probes, probe_dist=probe_dist,
                normalize_probes=normalize_probes, orthogonalize=orthogonalize, normalize_curl=normalize_curl,
                penalty_train_flag=penalty_train_flag, pool_factor=pool_factor, pool_to=pool_to, enforce_stride=enforce_stride,
            )

            grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
            for p, g, st in zip(params, grads, states):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([p], grad_clip)
                _sghmc_update(p, g, st, sgmcmc_eta, sgmcmc_temperature, sgmcmc_friction, weight_decay)

            ema.update_parameters(model)
            step += 1
            if sgmcmc_collect and step >= sgmcmc_burnin_steps and (step - sgmcmc_burnin_steps) % sgmcmc_thin == 0:
                snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                posterior.append(snapshot)
                if len(posterior) >= sgmcmc_max_samples:
                    break
        if sgmcmc_collect and len(posterior) >= sgmcmc_max_samples:
            break
    return (ema, posterior) if return_posterior else (ema, None)

@torch.no_grad()
def posterior_sampler(model, posterior_states, x, t, y=None, n_samples=8, reduce='mean_var', device=None):
    device = device or next(model.parameters()).device
    x = x.to(device); t = t.to(device); y = y.to(device) if y is not None else None
    outs = []
    for i, state in enumerate(posterior_states[:n_samples]):
        model.load_state_dict(state, strict=False)
        outs.append(model(x, t, y, train=False))
    V = torch.stack(outs, dim=0)
    if reduce == 'mean_var':
        return V.mean(dim=0), V.var(dim=0, unbiased=False)
    elif reduce == 'raw':
        return V
    elif reduce == 'mean_std':
        return V.mean(dim=0), V.var(dim=0, unbiased=False).sqrt()
    else:
        raise ValueError("reduce must be one of {'mean_var','mean_std','raw'}")
