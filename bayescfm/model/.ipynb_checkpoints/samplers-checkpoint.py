from __future__ import annotations
import torch
from typing import Callable, Optional

def ode_rhs(model: UNetCFM, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
    return model(x, t, y, train=False)

def rk4(model: UNetCFM, x0: torch.Tensor, y: Optional[torch.Tensor], t0: float, t1: float, steps: int) -> torch.Tensor:
    """Fixed-step RK4 integrator."""
    device = x0.device
    x = x0
    h = (t1 - t0) / steps
    for i in range(steps):
        t = torch.full((x.size(0),), t0 + i * h, device=device)
        k1 = ode_rhs(model, x, t, y)
        k2 = ode_rhs(model, x + 0.5 * h * k1, t + 0.5 * h, y)
        k3 = ode_rhs(model, x + 0.5 * h * k2, t + 0.5 * h, y)
        k4 = ode_rhs(model, x + h * k3, t + h, y)
        x = x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

def rk45(model: UNetCFM, x0: torch.Tensor, y: Optional[torch.Tensor], t0: float, t1: float, rtol: float = 1e-4, atol: float = 1e-5, max_steps: int = 10_000):
    """
    Adaptive Dormand–Prince RK45 integrator (no external deps).
    Integrates batch states x in parallel, using a shared step size.
    """
    device = x0.device
    x = x0
    t = torch.full((x.size(0),), t0, device=device)
    h = torch.tensor(t1 - t0, device=device) / 32  # initial step guess

    # DP(5) coefficients
    a = [0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0]
    b = [
        [],
        [1/5],
        [3/40, 9/40],
        [44/45, -56/15, 32/9],
        [19372/6561, -25360/2187, 64448/6561, -212/729],
        [9017/3168,  -355/33,     46732/5247,  49/176,  -5103/18656],
    ]
    c_sol = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    c_err = [
        35/384 - 5179/57600,
        0,
        500/1113 - 7571/16695,
        125/192 - 393/640,
        -2187/6784 + 92097/339200,
        11/84 - 187/2100,
        -1/40,
    ]
    safety = 0.9
    min_h = (t1 - t0) / 1e6
    max_h = (t1 - t0) / 8

    steps = 0
    while (t < t1).any():
        h = torch.clamp(h, min=min_h, max=max_h)
        # Ensure we don't step past t1
        h = torch.where(t + h > t1, t1 - t, h)

        # k stages
        k = []
        t_stage = t
        x_stage = x
        # k1
        k1 = ode_rhs(model, x_stage, t_stage, y); k.append(k1)
        # k2..k6
        for i_stage in range(1, 6):
            ti = t + a[i_stage] * h
            xi = x
            for j in range(i_stage):
                xi = xi + h * b[i_stage][j] * k[j]
            ki = ode_rhs(model, xi, ti, y)
            k.append(ki)
        # Fifth-order solution
        x_5 = x
        for ci, ki in zip(c_sol, k):
            x_5 = x_5 + h * ci * ki
        # Error estimate (5th - 4th)
        err = torch.zeros_like(x)
        for ce, ki in zip(c_err, k + [ode_rhs(model, x_5, t + h, y)]):
            err = err + h * ce * ki
        # Compute error norm per sample (RMS over dims)
        scale = atol + torch.max(torch.abs(x), torch.abs(x_5)) * rtol
        err_norm = (err.pow(2) / (scale.pow(2) + 1e-12)).mean(dim=(1,2,3)).sqrt()
        err_max = err_norm.max()

        # Accept/reject
        accept = err_max <= 1.0
        if accept:
            x = x_5
            t = t + h

        # Adapt step
        if err_max == 0:
            factor = 2.0
        else:
            factor = safety * (1.0 / err_max).pow(0.2)  # ^(1/(order+1)) with order=4
        h = torch.clamp(h * factor, min=min_h, max=max_h)

        steps += 1
        if steps > max_steps:
            break

    return x

def sample_ode(
    model: UNetCFM,
    n: int,
    shape: Tuple[int,int,int],   # (C,H,W)
    *,
    y: Optional[torch.Tensor] = None,
    device: str = "cuda",
    solver: str = "rk45",
    steps: int = 100,            # for rk4
    rtol: float = 1e-4,
    atol: float = 1e-5,
):
    """
    Draw x0 ~ N(0,I), integrate dx/dt = v_theta(x,t,y) from t=0 -> 1.
    """
    model.eval().to(device)
    C, H, W = shape
    x0 = torch.randn(n, C, H, W, device=device)
    if y is not None:
        y = y.to(device)

    if solver == "rk4":
        x1 = rk4(model, x0, y, t0=0.0, t1=1.0, steps=steps)
    else:
        x1 = rk45(model, x0, y, t0=0.0, t1=1.0, rtol=rtol, atol=atol)
    return x1.clamp(-1, 1)

