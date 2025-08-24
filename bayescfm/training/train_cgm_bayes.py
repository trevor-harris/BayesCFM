import math
import torch
from ..losses.cfm import cgm_total_loss_fd
from ..model.unet import UNetCFM

def _apply_weight_decay_to_grad(param, grad, weight_decay):
    if weight_decay and weight_decay != 0.0:
        # Equivalent to adding L2 prior: ∇U ← ∇U + λ θ
        grad = grad.add(param, alpha=weight_decay)
    return grad

def _sgld_update(param, grad, *, eta, temperature, weight_decay):
    """
    SGLD: θ ← θ - (η/2) ∇U(θ) + sqrt(2 η T) ξ
    All parameter writes are done under no_grad to avoid leaf in-place errors.
    """
    g = _apply_weight_decay_to_grad(param, grad, weight_decay)
    noise_std = math.sqrt(2.0 * eta * temperature)
    with torch.no_grad():
        param.add_(g, alpha=-0.5 * eta)
        param.add_(torch.randn_like(param) * noise_std)

class _SGHMCState:
    def __init__(self, param):
        # momentum buffer must not require grad
        self.v = torch.zeros_like(param, requires_grad=False)

def _sghmc_update(param, grad, state: _SGHMCState, *, eta, temperature, friction, weight_decay):
    """
    SGHMC:
      v ← (1 - α) v - η ∇U(θ) + sqrt(2 α η T) ξ
      θ ← θ + v
    """
    g = _apply_weight_decay_to_grad(param, grad, weight_decay)
    alpha = friction
    noise_std = math.sqrt(2.0 * alpha * eta * temperature)
    with torch.no_grad():
        state.v.mul_(1.0 - alpha).add_(g, alpha=-eta).add_(torch.randn_like(param) * noise_std)
        param.add_(state.v)

def train_cgm_bayes(
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

    # --- Finite-difference controls ---
    probes: int = 1,
    eps_scale: float = 5e-4,
    normalize_probes: bool = True,
    central_mono: bool = False,

    # --- Penalty pooling ---
    pool_factor: int | None = None,
    pool_to: tuple[int, int] | None = None,
    enforce_stride: bool = True,

    # --- Stability knobs for penalty ---
    curl_central: bool = True,
    probe_dist: str = "rademacher",
    orthogonalize: bool = True,
    penalty_train_flag: bool = False,
    normalize_curl: bool = True,
    huber_delta: float = 1.0,
    delta: float = 1e-8,
    probe_seed: int | None = None,

    # --- SG-MCMC knob ---
    sgmcmc_enable: bool = False,            # turn on SG-MCMC updates
    sgmcmc_alg: str = "sgld",               # {"sgld","sghmc"}
    sgmcmc_temperature: float = 1.0,        # T
    sgmcmc_eta: float | None = None,        # step size for SG-MCMC (defaults to lr if None)
    sgmcmc_friction: float = 0.05,          # α for SGHMC (ignored for SGLD)

    # --- Posterior collection during training ---
    sgmcmc_collect: bool = False,           # collect posterior snapshots during training
    sgmcmc_burnin_steps: int = 0,           # number of steps to skip before collecting
    sgmcmc_thin: int = 100,                 # collect every 'thin' steps
    sgmcmc_max_samples: int = 20,           # max snapshots to collect
    return_posterior: bool = False,         # if True, also return collected snapshots
):
    """
    If sgmcmc_enable=False: uses AdamW like before.
    If sgmcmc_enable=True: replaces optimizer with SG-MCMC updates (SGLD/SGHMC).
    Optionally collects posterior snapshots (state_dict CPU copies) during training.
    """
    model.to(device).train()

    # Standard optimizer (only used if SG-MCMC is disabled)
    opt = None if sgmcmc_enable else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # SGHMC momentum buffers (if needed)
    sghmc_states = None
    if sgmcmc_enable and sgmcmc_alg.lower() == "sghmc":
        sghmc_states = {p: _SGHMCState(p) for p in model.parameters() if p.requires_grad}

    # EMA params (still useful as a smoothed predictor)
    ema_params = [p.detach().clone() for p in model.parameters()]

    # Posterior snapshots
    posterior_states: list[dict] = []
    eta = float(sgmcmc_eta if (sgmcmc_enable and sgmcmc_eta is not None) else lr)

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

            loss, logs = cgm_total_loss_fd(
                model, x1, t, labels,
                lambda_curl=lambda_curl, lambda_mono=lambda_mono,
                probes=probes, eps_scale=eps_scale, normalize_probes=normalize_probes,
                central_mono=central_mono,
                pool_factor=pool_factor, pool_to=pool_to, enforce_stride=enforce_stride,
                curl_central=curl_central, probe_dist=probe_dist, orthogonalize=orthogonalize,
                penalty_train_flag=penalty_train_flag,
                normalize_curl=normalize_curl, huber_delta=huber_delta, delta=delta,
                probe_seed=probe_seed,
            )

            # Backprop
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if not sgmcmc_enable:
                # Standard AdamW update
                opt.step()
            else:
                # SG-MCMC update (in-place)
                alg = sgmcmc_alg.lower()
                if alg == "sgld":
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        _sgld_update(
                            p, p.grad,
                            eta=eta, temperature=sgmcmc_temperature,
                            weight_decay=weight_decay
                        )
                elif alg == "sghmc":
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        _sghmc_update(
                            p, p.grad, sghmc_states[p],
                            eta=eta, temperature=sgmcmc_temperature,
                            friction=sgmcmc_friction,
                            weight_decay=weight_decay
                        )
                else:
                    raise ValueError(f"Unknown sgmcmc_alg: {sgmcmc_alg}")

            # EMA (always computed)
            with torch.no_grad():
                for ep, p in zip(ema_params, model.parameters()):
                    ep.mul_(ema_decay).add_(p, alpha=1 - ema_decay)

            # Posterior collection
            if sgmcmc_enable and sgmcmc_collect:
                if step >= sgmcmc_burnin_steps and (step - sgmcmc_burnin_steps) % max(1, sgmcmc_thin) == 0:
                    if len(posterior_states) < sgmcmc_max_samples:
                        # Store CPU copy to save GPU memory
                        posterior_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

            step += 1
            if step % log_every == 0:
                print(
                    f"[epoch {epoch}] step {step:>7d} "
                    f"total={logs['loss_total']:.4f}  "
                    f"cfm={logs['loss_cfm']:.4f}  "
                    f"curl={logs['reg_curl']:.4f}  "
                    f"curl_raw={logs['reg_curl_raw']:.4f}  "
                    f"mono={logs['reg_mono']:.4f}"
                    + (f"  (SG-MCMC {sgmcmc_alg}, eta={eta:.2e}, T={sgmcmc_temperature})" if sgmcmc_enable else "")
                )

    # Build EMA model (same config as model)
    ema_model = UNetCFM(**{k: getattr(model, k) for k in [
        "in_channels","model_channels","out_channels","channel_mult","num_res_blocks",
        "attn_resolutions","num_heads","dropout","num_classes","class_dropout_prob","groupnorm_groups"
    ]})
    ema_model.load_state_dict({k: v for (k,_), v in zip(model.state_dict().items(), ema_params)})
    ema_model.to(device).eval()

    if return_posterior:
        return ema_model, posterior_states
    return ema_model

