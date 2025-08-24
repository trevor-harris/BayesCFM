from __future__ import annotations
import torch
from ..losses.cfm import cgm_total_loss_ad
from ..model.unet import UNetCFM

def train_cgm(
    model,
    dataloader,
    *,
    epochs: int = 100,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    device: str = "cuda",
    ema_decay: float = 0.999,
    max_grad_norm: float | None = 1.0,
    log_every: int = 100,

    # --- Regularizer strengths ---
    lambda_curl: float = 1e-4,
    lambda_mono: float = 1e-4,

    # --- autodiff controls ---
    probes: int = 1,
    # eps_scale: float = 5e-4,
    normalize_probes: bool = True,
    # central_mono: bool = False,       # forward diff for monotonicity by default

    # --- Penalty pooling (keep training loss at full res) ---
    pool_factor: int | None = None,   # e.g., 2 -> H/2 x W/2
    pool_to: tuple[int, int] | None = None,  # e.g., (96,96)
    enforce_stride: bool = True,

    # --- Stability knobs for curl/penalty pass ---
    # curl_central: bool = True,        # central diff for curl (recommended)
    probe_dist: str = "rademacher",   # {"rademacher","gaussian"}
    orthogonalize: bool = True,       # make s ⟂ r
    penalty_train_flag: bool = False, # disable dropout/CFG in penalty forwards
    normalize_curl: bool = True,      # scale-invariant + robust curl
    # huber_delta: float = 1.0,         # Huber δ for normalized curl
    # delta: float = 1e-8,              # small numerical floor
    # probe_seed: int | None = None,    # deterministic probes per call if set
):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema_params = [p.detach().clone() for p in model.parameters()]

    step = 0
    for epoch in range(1, epochs + 1):
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x1, labels = batch
            else:
                x1, labels = batch["images"], batch.get("labels", None)

            x1 = x1.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)

            B = x1.size(0)
            t = torch.rand(B, device=device)

            loss, logs = cgm_total_loss_ad(
                model, x1, t, labels,
                lambda_curl=lambda_curl, 
                lambda_mono=lambda_mono,
                probes=probes, 
                normalize_probes=normalize_probes,
                pool_factor=pool_factor, 
                pool_to=pool_to, 
                enforce_stride=enforce_stride,
                probe_dist=probe_dist,
                orthogonalize=orthogonalize,
                penalty_train_flag=penalty_train_flag,
                normalize_curl=normalize_curl,
                # huber_delta=huber_delta, 
                # delta=delta,
                # probe_seed=probe_seed,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            # EMA
            with torch.no_grad():
                for ep, p in zip(ema_params, model.parameters()):
                    ep.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            step += 1
            if step % log_every == 0:
                print(
                    f"[epoch {epoch}] step {step:>7d} "
                    f"total={logs['loss_total']:.4f}  "
                    f"cfm={logs['loss_cfm']:.4f}  "
                    f"curl={logs['reg_curl']:.4f}  "
                    # f"curl_raw={logs['reg_curl_raw']:.4f}  "
                    f"mono={logs['reg_mono']:.4f}"
                )

    # Return EMA copy for eval (same config as model)
    ema_model = UNetCFM(**{k: getattr(model, k) for k in [
        "in_channels","model_channels","out_channels","channel_mult","num_res_blocks",
        "attn_resolutions","num_heads","dropout","num_classes","class_dropout_prob","groupnorm_groups"
    ]})
    ema_model.load_state_dict({k: v for (k,_), v in zip(model.state_dict().items(), ema_params)})
    ema_model.to(device).eval()
    return ema_model

