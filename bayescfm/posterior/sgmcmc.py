import torch
from typing import Optional

def posterior_sampler(
    model: torch.nn.Module,
    posterior_states: list[dict],
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    n_samples: int = 1,
    reduce: str = "none",      # {"none","mean","var","mean_var"}
    device: str | None = None,
):
    """
    Draws velocity-field samples v(x,t,y) from SG-MCMC posterior snapshots.

    Args:
        model: a UNetCFM instance (weights will be swapped temporarily)
        posterior_states: list of state_dict snapshots (CPU tensors ok)
        x, t, y: inputs
        n_samples: number of posterior draws to evaluate
        reduce:  "none" -> return list of tensors
                 "mean" -> return mean over draws
                 "var"  -> return variance over draws
                 "mean_var" -> return (mean, var)
        device: move model & data to this device temporarily (defaults to x.device)

    Returns:
        Depending on reduce:
          - list[Tensor] of length n_samples with shape [B,C,H,W]
          - or a single Tensor (mean/var), or tuple (mean, var)
    """
    assert len(posterior_states) > 0, "posterior_states is empty."
    device = device or x.device

    # Save original weights to restore after sampling
    orig = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model = model.to(device).eval()

    draws = []
    for i in range(n_samples):
        sd = posterior_states[i % len(posterior_states)]
        model.load_state_dict(sd, strict=True)
        v = model(x.to(device), t.to(device), None if y is None else y.to(device), train=False)
        draws.append(v.detach().cpu())

    # Restore original weights
    model.load_state_dict(orig, strict=True)

    if reduce == "none":
        return draws
    stacked = torch.stack(draws, dim=0)  # [S,B,C,H,W]
    if reduce == "mean":
        return stacked.mean(0)
    if reduce == "var":
        return stacked.var(0, unbiased=False)
    if reduce == "mean_var":
        return stacked.mean(0), stacked.var(0, unbiased=False)
    raise ValueError(f"Unknown reduce: {reduce}")

