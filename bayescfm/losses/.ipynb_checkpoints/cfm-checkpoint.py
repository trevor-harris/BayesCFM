from __future__ import annotations
import torch, torch.nn.functional as F
from typing import Optional, Tuple

def _linear_schedule(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    s = t[:, None, None, None]
    sdot = torch.ones_like(s)
    return s, sdot

def cfm_loss_ot(model, x1: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
    s, sdot = _linear_schedule(t)
    x0 = torch.randn_like(x1)
    xt = (1 - s) * x0 + s * x1
    v_star = sdot * (x1 - x0)
    v_pred = model(xt, t, y, train=True)
    return F.mse_loss(v_pred, v_star)

@torch.no_grad()
def _unitize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    n = v.flatten(1).norm(dim=1, keepdim=True).clamp_min(eps)
    return v / n.view(v.size(0), *([1] * (v.dim()-1)))

@torch.no_grad()
def _rademacher_like(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x).bernoulli_(0.5).mul_(2.0).add_(-1.0)

def _sample_probe(shape_like: torch.Tensor, dist: str = "rademacher", normalize: bool = True) -> torch.Tensor:
    if dist.lower() == "rademacher":
        r = _rademacher_like(shape_like)
    elif dist.lower() == "gaussian":
        r = torch.randn_like(shape_like)
    else:
        raise ValueError(f"unknown probe_dist={dist}")
    if normalize:
        r = _unitize(r)
    return r

@torch.no_grad()
def _orthogonalize_s_against_r(r: torch.Tensor, s: torch.Tensor, renorm: bool = True) -> torch.Tensor:
    r_flat = r.flatten(1)
    s_flat = s.flatten(1)
    rs = (r_flat * s_flat).sum(dim=1, keepdim=True)
    rr = (r_flat * r_flat).sum(dim=1, keepdim=True).clamp_min(1e-12)
    s_flat = s_flat - rs / rr * r_flat
    s_new = s_flat.view_as(s)
    if renorm: s_new = _unitize(s_new)
    return s_new

def _inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).flatten(1).sum(dim=1)

def _snap_to_stride(h: int, w: int, stride: int):
    H = max(stride, (h // stride) * stride)
    W = max(stride, (w // stride) * stride)
    return H, W

def _jvp(model, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor], r: torch.Tensor, train_flag: bool):
    x = x.requires_grad_(True)
    def f(inp): return model(inp, t, y, train=train_flag)
    y_out, jvp_out = torch.autograd.functional.jvp(f, (x,), (r,), create_graph=True, strict=True)
    return y_out, jvp_out

def gradient_field_penalty_ad(
    model, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
    *, probes: int = 1, probe_dist: str = "rademacher", normalize_probes: bool = True,
    orthogonalize: bool = True, normalize_curl: bool = True, penalty_train_flag: bool = True,
    pool_factor: int | None = None, pool_to: tuple[int,int] | None = None, enforce_stride: bool = True,
):
    B, C, H, W = x.shape
    if pool_to is not None:
        Ht, Wt = pool_to
        if enforce_stride:
            downs = max(0, len(getattr(model, "channel_mult", (1,))) - 1)
            stride = 2 ** downs
            Ht, Wt = _snap_to_stride(Ht, Wt, stride)
        x_work = torch.nn.functional.adaptive_avg_pool2d(x, (Ht, Wt))
    elif pool_factor is not None and pool_factor > 1:
        x_work = torch.nn.functional.avg_pool2d(x, kernel_size=pool_factor, stride=pool_factor)
    else:
        x_work = x

    curl_acc = x.new_zeros(())
    mono_acc = x.new_zeros(())
    for _ in range(probes):
        r = _sample_probe(x_work, dist=probe_dist, normalize=normalize_probes)
        s = _sample_probe(x_work, dist=probe_dist, normalize=normalize_probes)
        if orthogonalize: s = _orthogonalize_s_against_r(r, s, renorm=normalize_probes)
        _, Jr = _jvp(model, x_work, t, y, r, train_flag=penalty_train_flag)
        _, Js = _jvp(model, x_work, t, y, s, train_flag=penalty_train_flag)
        a_rs = _inner(Js, r) - _inner(Jr, s)
        if normalize_curl:
            denom = (r.flatten(1).pow(2).sum(dim=1) + s.flatten(1).pow(2).sum(dim=1)).sqrt().clamp_min(1e-12)
            a_rs = a_rs / denom
        curl_acc = curl_acc + a_rs.pow(2).mean()
        rtJr = _inner(Jr, r)
        mono_acc = mono_acc + torch.nn.functional.softplus(-rtJr).mean()
    return {"curl": curl_acc / probes, "monotonicity": mono_acc / probes, "total": (curl_acc + mono_acc) / probes}

def cgm_total_loss_ad(
    model, x1: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, *,
    lambda_curl: float = 0.0, lambda_mono: float = 0.0, probes: int = 1, probe_dist: str = "rademacher",
    normalize_probes: bool = True, orthogonalize: bool = True, normalize_curl: bool = True, penalty_train_flag: bool = True,
    pool_factor: int | None = None, pool_to: tuple[int,int] | None = None, enforce_stride: bool = True,
):
    s = t[:, None, None, None]; sdot = torch.ones_like(s)
    x0 = torch.randn_like(x1)
    xt = (1 - s) * x0 + s * x1
    v_star = sdot * (x1 - x0)
    v_pred = model(xt, t, y, train=True)
    loss_cfm = torch.nn.functional.mse_loss(v_pred, v_star)
    reg = gradient_field_penalty_ad(
        model, xt, t, y, probes=probes, probe_dist=probe_dist,
        normalize_probes=normalize_probes, orthogonalize=orthogonalize, normalize_curl=normalize_curl,
        penalty_train_flag=penalty_train_flag, pool_factor=pool_factor, pool_to=pool_to, enforce_stride=enforce_stride,
    )
    loss = loss_cfm + lambda_curl * reg["curl"] + lambda_mono * reg["monotonicity"]
    logs = {"loss_total": loss.detach(), "loss_cfm": loss_cfm.detach(), "reg_curl": reg["curl"].detach(), "reg_mono": reg["monotonicity"].detach()}
    return loss, logs
