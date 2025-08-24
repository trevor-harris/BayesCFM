import torch
import torch.nn.functional as F

def _linear_schedule(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """s(t)=t, sdot=1"""
    return t, torch.ones_like(t)

def cfm_loss_ot(
    model: UNetCFM,
    x1: torch.Tensor,                # real data in [-1,1]
    t: torch.Tensor,                 # [B] ~ U(0,1)
    y: Optional[torch.Tensor] = None # labels if class-cond
) -> torch.Tensor:
    """
    OT conditional path: x_t = (1-s(t)) x0 + s(t) x1,  v* = s'(t) (x1 - x0)
    x0 ~ N(0, I).
    """
    B = x1.shape[0]
    s, sdot = _linear_schedule(t)  # easy default
    s = s[:, None, None, None]
    sdot = sdot[:, None, None, None]

    x0 = torch.randn_like(x1)
    xt = (1 - s) * x0 + s * x1
    v_star = sdot * (x1 - x0)

    v_pred = model(xt, t, y, train=True)
    loss = F.mse_loss(v_pred, v_star)
    return loss

def _unitize(v: torch.Tensor, eps: float = 1e-12):
    n = v.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)
    return v / n.view(v.size(0), 1, 1, 1)

def _inner(a: torch.Tensor, b: torch.Tensor):
    return (a * b).flatten(1).sum(dim=1)

def _snap_to_stride(h: int, w: int, stride: int) -> tuple[int, int]:
    # Make (h,w) multiples of 'stride' so the U-Net down/upsample path matches.
    H = max(stride, (h // stride) * stride)
    W = max(stride, (w // stride) * stride)
    return H, W

def _sample_probe(shape, *, device, dtype, dist: str, generator=None):
    if dist == "rademacher":
        r = torch.randint(0, 2, shape, device=device, generator=generator, dtype=torch.int8)
        r = r.to(dtype=dtype) * 2 - 1  # {0,1} -> {-1,+1}
        return r
    elif dist == "gaussian":
        return torch.randn(shape, device=device, generator=generator, dtype=dtype)
    else:
        raise ValueError(f"Unknown probe_dist: {dist}")

def _orthogonalize_s_against_r(s, r, eps: float = 1e-12):
    # Per-sample projection: s <- s - proj_r(s)
    num = _inner(s, r)                        # [B]
    den = _inner(r, r).clamp_min(eps)         # [B]
    s = s - (num / den)[:, None, None, None] * r
    return s

def gradient_field_penalty_fd(
    model,
    x: torch.Tensor,           # [B,C,H,W]
    t: torch.Tensor,           # [B]
    y: torch.Tensor | None = None,
    *,
    probes: int = 1,
    eps_scale: float = 1e-3,
    normalize_probes: bool = True,
    central_mono: bool = False,

    # Pooling options (kept as-is)
    pool_factor: int | None = None,          # e.g., 2 => H/2 x W/2 via avg pooling
    pool_to: tuple[int, int] | None = None,  # e.g., (64,64) via adaptive avg pooling
    enforce_stride: bool = True,             # snap to multiples of 2^(#downs) if True
    reuse_fx: torch.Tensor | None = None,    # (unused when curl_central=True)

    # NEW stability knobs
    curl_central: bool = True,               # central differences for curl Δr, Δs
    probe_dist: str = "rademacher",          # {"rademacher","gaussian"}
    orthogonalize: bool = True,              # make s ⟂ r, then re-normalize
    penalty_train_flag: bool = False,        # call model(..., train=False) in penalty pass
    normalize_curl: bool = True,             # divide curl scalar by a local scale
    huber_delta: float = 1.0,                # Huber delta for curl (if normalize_curl)
    delta: float = 1e-8,                     # small numerical floor in denominators
    probe_seed: int | None = None,           # make probes deterministic per-call if set
):
    """
    Forward-only finite differences to approximate:
      - Curl/antisymmetry via scalar Hutchinson (Δ_s, Δ_r)
      - Monotonicity via r^T J r ≈ <Δ_r, r>

    Optional downsampling:
      - pool_factor=k uses F.avg_pool2d(x, k, k)
      - pool_to=(H,W) uses F.adaptive_avg_pool2d to (H,W) (snapped to valid stride)

    Stability additions:
      - Rademacher probes (default), orthogonalize s against r, central diffs for curl,
        normalized & robust (Huber) curl scalar, deterministic probes, and penalty
        forwards with train=False to remove dropout/CFG noise.
    """
    device, dtype = x.device, x.dtype
    B, C, H, W = x.shape

    # Decide working resolution for the penalty (unchanged)
    if pool_to is not None:
        Ht, Wt = pool_to
        if enforce_stride:
            downs = max(0, len(getattr(model, "channel_mult", (1,))) - 1)
            stride = 2 ** downs
            Ht, Wt = _snap_to_stride(Ht, Wt, stride)
        x_work = F.adaptive_avg_pool2d(x, (Ht, Wt))
    elif pool_factor is not None and pool_factor > 1:
        assert H % pool_factor == 0 and W % pool_factor == 0, \
            "pool_factor must divide H and W; otherwise use pool_to=(H',W')"
        x_work = F.avg_pool2d(x, kernel_size=pool_factor, stride=pool_factor)
    else:
        x_work = x

    eps = torch.tensor(eps_scale, device=device, dtype=dtype)

    # Base forward at working resolution (train flag configurable)
    if reuse_fx is None or curl_central:
        fx0 = model(x_work, t, y, train=penalty_train_flag)
    else:
        fx0 = reuse_fx

    curl_acc = x.new_zeros(())
    mono_acc = x.new_zeros(())
    curl_raw_acc = x.new_zeros(())

    # Deterministic probe generator if requested
    gen = None
    if probe_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(probe_seed))

    for _ in range(probes):
        # Probes r, s
        r = _sample_probe(x_work.shape, device=device, dtype=dtype, dist=probe_dist, generator=gen)
        s = _sample_probe(x_work.shape, device=device, dtype=dtype, dist=probe_dist, generator=gen)

        if orthogonalize:
            s = _orthogonalize_s_against_r(s, r)
        if normalize_probes:
            r = _unitize(r)
            s = _unitize(s)

        # ---- Monotonicity (optionally central; often ok with forward) ----
        fx_r_plus = model(x_work + eps * r, t, y, train=penalty_train_flag)
        if central_mono:
            fx_r_minus = model(x_work - eps * r, t, y, train=penalty_train_flag)
            Jr_dir = (fx_r_plus - fx_r_minus) / (2 * eps)
        else:
            Jr_dir = (fx_r_plus - fx0) / eps
        rtJr = _inner(Jr_dir, r)  # [B]
        mono_acc = mono_acc + F.softplus(-rtJr).mean()

        # ---- Curl antisymmetry scalar (stabilized) -----------------------
        if curl_central:
            fx_r_minus = model(x_work - eps * r, t, y, train=penalty_train_flag)
            fx_s_plus  = model(x_work + eps * s, t, y, train=penalty_train_flag)
            fx_s_minus = model(x_work - eps * s, t, y, train=penalty_train_flag)
            delta_r = (fx_r_plus - fx_r_minus) / (2 * eps)
            delta_s = (fx_s_plus - fx_s_minus) / (2 * eps)
        else:
            fx_s_plus = model(x_work + eps * s, t, y, train=penalty_train_flag)
            delta_r = (fx_r_plus - fx0) / eps
            delta_s = (fx_s_plus - fx0) / eps

        a_rs = _inner(delta_s, r) - _inner(delta_r, s)  # [B]
        curl_raw_acc = curl_raw_acc + (a_rs.pow(2).mean())

        if normalize_curl:
            # scale-invariant normalization + robust Huber
            nr = delta_r.flatten(1).norm(dim=1)
            ns = delta_s.flatten(1).norm(dim=1)
            den = (0.5 * (nr + ns)).clamp_min(delta)      # [B]
            a_norm = a_rs / den
            curl_term = F.huber_loss(a_norm, torch.zeros_like(a_norm), delta=huber_delta, reduction="mean")
        else:
            curl_term = a_rs.pow(2).mean()

        curl_acc = curl_acc + curl_term

    curl = curl_acc / probes
    monotonicity = mono_acc / probes
    curl_raw = curl_raw_acc / probes  # for logging
    return {"curl": curl, "monotonicity": monotonicity, "curl_raw": curl_raw, "total": curl + monotonicity}

def cgm_total_loss_fd(
    model,
    x1: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None,
    *,
    lambda_curl: float = 0.0,
    lambda_mono: float = 0.0,
    probes: int = 1,
    eps_scale: float = 1e-3,
    normalize_probes: bool = True,
    central_mono: bool = False,

    # Pooling (kept)
    pool_factor: int | None = None,
    pool_to: tuple[int, int] | None = None,
    enforce_stride: bool = True,

    # NEW stability knobs (passed through)
    curl_central: bool = True,
    probe_dist: str = "rademacher",
    orthogonalize: bool = True,
    penalty_train_flag: bool = False,
    normalize_curl: bool = True,
    huber_delta: float = 1.0,
    delta: float = 1e-8,
    probe_seed: int | None = None,
):
    # OT path with s(t)=t
    s = t[:, None, None, None]
    sdot = torch.ones_like(s)
    x0 = torch.randn_like(x1)
    xt = (1 - s) * x0 + s * x1
    v_star = sdot * (x1 - x0)

    # CFM loss at full resolution
    v_pred = model(xt, t, y, train=True)
    loss_cfm = F.mse_loss(v_pred, v_star)

    # Penalty (optionally pooled; deterministic/stable options available)
    reg = gradient_field_penalty_fd(
        model, xt.detach(), t.detach(), y.detach() if y is not None else None,
        probes=probes, eps_scale=eps_scale, normalize_probes=normalize_probes,
        central_mono=central_mono,
        pool_factor=pool_factor, pool_to=pool_to, enforce_stride=enforce_stride,
        curl_central=curl_central, probe_dist=probe_dist, orthogonalize=orthogonalize,
        penalty_train_flag=penalty_train_flag,
        normalize_curl=normalize_curl, huber_delta=huber_delta, delta=delta,
        probe_seed=probe_seed,
    )

    loss = loss_cfm + lambda_curl * reg["curl"] + lambda_mono * reg["monotonicity"]
    logs = {
        "loss_total": loss.detach(),
        "loss_cfm": loss_cfm.detach(),
        "reg_curl": reg["curl"].detach(),
        "reg_curl_raw": reg["curl_raw"].detach(),
        "reg_mono": reg["monotonicity"].detach(),
    }
    return loss, logs

