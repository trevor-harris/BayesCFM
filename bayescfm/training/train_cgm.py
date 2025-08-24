
import torch
from ..losses.cfm import cgm_total_loss_ad

def train_cgm(
    model, loader, *, epochs: int = 1, lr: float = 2e-4, device="cuda",
    weight_decay: float = 0.0,
    lambda_curl: float = 0.0, lambda_mono: float = 0.0,
    probes: int = 1, probe_dist: str = "rademacher",
    normalize_probes: bool = True, orthogonalize: bool = True, normalize_curl: bool = True,
    penalty_train_flag: bool = True,
    pool_factor=None, pool_to=None, enforce_stride: bool = True,
    grad_clip: float = 0.0, ema_decay: float = 0.999,
):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = torch.optim.swa_utils.AveragedModel(model)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x1 = batch[0].to(device)
                y = batch[1].to(device) if len(batch) > 1 else None
            elif isinstance(batch, dict):
                x1 = batch.get("images").to(device)
                y = batch.get("labels", None)
                y = y.to(device) if y is not None else None
            else:
                x1, y = batch.to(device), None

            B = x1.size(0)
            t = torch.rand(B, device=device)

            loss, logs = cgm_total_loss_ad(
                model, x1, t, y,
                lambda_curl=lambda_curl, lambda_mono=lambda_mono,
                probes=probes, probe_dist=probe_dist,
                normalize_probes=normalize_probes, orthogonalize=orthogonalize, normalize_curl=normalize_curl,
                penalty_train_flag=penalty_train_flag,
                pool_factor=pool_factor, pool_to=pool_to, enforce_stride=enforce_stride,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            ema.update_parameters(model)
    return ema
