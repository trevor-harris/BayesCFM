import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

def eval_mode(model: nn.Module):
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()

def _ot_path_targets(x1: torch.Tensor, t: torch.Tensor):
    """OT conditional path: x_t = (1-t) x0 + t x1, v* = (x1-x0) with s(t)=t."""
    s = t[:, None, None, None]
    x0 = torch.randn_like(x1)
    xt = (1 - s) * x0 + s * x1
    v_star = (x1 - x0)  # since sdot=1
    return xt, v_star

def _module_list(model: nn.Module, mode: str):
    """Select modules to include in Laplace. mode: 'last_layer' or 'kfac_all'."""
    if mode == "last_layer":
        # assumes your UNetCFM naming (head is a Conv2d)
        return [model.head]
    elif mode == "kfac_all":
        mods = []
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                mods.append(m)
        return mods
    else:
        raise ValueError(f"Unknown mode: {mode}")

def _add_bias_row(patches: torch.Tensor) -> torch.Tensor:
    """Append a row of ones to activation patches: [B, Din, HW] -> [B, Din+1, HW]."""
    B, Din, HW = patches.shape
    ones = patches.new_ones(B, 1, HW)
    return torch.cat([patches, ones], dim=1)

def _chol_precision_solve(L: torch.Tensor, B: torch.Tensor, upper: bool = False) -> torch.Tensor:
    """
    Solve (L L^T)^{-1/2} application via triangular solves:
    Given precision P = L L^T, to apply P^{-1/2} on right/left we use:
      y = solve(L, Z)  (lower)
      y = solve(L^T, Z) (upper)
    Here we expose the primitive 'solve L y = B'.
    """
    return torch.linalg.solve_triangular(L, B, upper=upper)

class _KFACStats:
    __slots__ = ("A_sum", "G_sum", "N")
    def __init__(self, A_dim, G_dim, device):
        self.A_sum = torch.zeros(A_dim, A_dim, device=device)
        self.G_sum = torch.zeros(G_dim, G_dim, device=device)
        self.N = 0  # total number of spatial locations aggregated

class _HookPack:
    def __init__(self):
        self.fwd_act = {}   # module -> activation (input)
        self.bwd_grad = {}  # module -> grad wrt pre-activation (grad_output[0])

class LaplaceCFM:
    """
    Laplace approximation (last-layer or KFAC over all layers) for your trained UNetCFM.

    Parameters
    ----------
    model : nn.Module
        Trained model (evaluated around its current weights: MAP point).
    mode : str
        'last_layer' (default) or 'kfac_all'.
    sigma_noise : float
        Observation noise std for Gaussian likelihood on velocity field.
    damping : float
        Factor damping for A and G (Tikhonov).
    device : str
        Device for accumulation (CPU/GPU). Use GPU for speed, CPU for memory.

    Usage
    -----
    lap = LaplaceCFM(model, mode='last_layer', sigma_noise=1.0, damping=1e-3)
    lap.fit(dataloader, steps=200)            # accumulate KFAC stats
    sd = lap.sample_state_dict()              # draw one posterior weight sample
    v_mean, v_var = lap.posterior_predict(x, t, y, n_samples=10, reduce='mean_var')
    """
    def __init__(self, model: nn.Module, mode: str = "last_layer",
                 sigma_noise: float = 1.0, damping: float = 1e-3, device: str = "cuda"):
        self.model = model
        self.mode = mode
        self.sigma = float(sigma_noise)
        self.damping = float(damping)
        self.device = device

        # which modules we approximate
        self.modules = _module_list(model, mode)

        # storage for KFAC factors per module
        self.stats: dict[nn.Module, _KFACStats] = {}
        # store layer metadata to reconstruct conv unfold shapes
        self.meta = {}

        # filled after fit()
        self.A_factors = {}
        self.G_factors = {}
        self.scales = {}               # scale per module: data_size / sigma^2
        self.Lr = {}
        self.Lc = {}

        # cache original weights for restoration after sampling
        self._orig_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # --------------- hooks registration ----------------

    def _register_hooks(self):
        hp = _HookPack()
        handles = []

        def fwd_hook(mod, inp, out):
            # store the module INPUT activation (pre-weights)
            hp.fwd_act[mod] = inp[0].detach()

        def bwd_hook(mod, grad_input, grad_output):
            # grad_output[0] is grad wrt module OUTPUT (pre-activation for Conv/Linear)
            hp.bwd_grad[mod] = grad_output[0].detach()

        for m in self.modules:
            handles.append(m.register_forward_hook(fwd_hook))
            handles.append(m.register_full_backward_hook(bwd_hook))
        return hp, handles

    # --------------- factor accumulation ----------------

    @torch.no_grad()
    def _accumulate_batch(self, hp: _HookPack):
        """
        After a forward/backward pass (loss.backward()), use captured
        activations & output grads to update KFAC stats for selected modules.
        """
        for m in self.modules:
            act = hp.fwd_act.get(m, None)
            gout = hp.bwd_grad.get(m, None)
            if (act is None) or (gout is None):
                continue

            if isinstance(m, nn.Conv2d):
                # unfold input activation into patches (B, Cin*k*k, HW)
                B, Cin, Hin, Win = act.shape
                kH, kW = m.kernel_size
                sH, sW = m.stride
                pH, pW = m.padding
                dH, dW = m.dilation
                patches = F.unfold(act, kernel_size=(kH, kW), dilation=(dH, dW),
                                   padding=(pH, pW), stride=(sH, sW))  # [B, Cin*k*k, Hout*Wout]
                if m.bias is not None:
                    patches = _add_bias_row(patches)  # [B, Din(+1), HW]
                Din = patches.size(1)
                # grads wrt pre-activation (module output), reshape to [B, Cout, HW]
                B2, Cout, Hout, Wout = gout.shape
                assert B2 == B
                g = gout.reshape(B, Cout, Hout * Wout)

                # covariances (mean over samples & positions)
                # A = E[a a^T], a: [Din, HW]
                # G = E[g g^T], g: [Cout, HW]
                A_sum = torch.zeros(Din, Din, device=patches.device)
                G_sum = torch.zeros(Cout, Cout, device=g.device)
                # accumulate over batch
                A_sum += (patches @ patches.transpose(1, 2)).sum(dim=0)  # sum over B
                G_sum += (g @ g.transpose(1, 2)).sum(dim=0)              # sum over B
                N = B * (Hout * Wout)

                if m not in self.stats:
                    self.stats[m] = _KFACStats(Din, Cout, device=patches.device)
                    # store conv meta so we can rebuild mapping if needed
                    self.meta[m] = {"is_conv": True, "Din": Din, "Cout": Cout}
                st = self.stats[m]
                st.A_sum += A_sum
                st.G_sum += G_sum
                st.N += N

            elif isinstance(m, nn.Linear):
                # activation: [B, Din], add bias if present
                a = act
                if m.bias is not None:
                    a = torch.cat([a, a.new_ones(a.size(0), 1)], dim=1)  # [B, Din+1]
                Din = a.size(1)
                # gradient wrt pre-activation: [B, Dout]
                g = hp.bwd_grad[m]  # [B, Dout]
                A_sum = a.t().mm(a)   # [Din, Din]
                G_sum = g.t().mm(g)   # [Dout, Dout]
                N = a.size(0)

                if m not in self.stats:
                    self.stats[m] = _KFACStats(Din, m.out_features, device=a.device)
                    self.meta[m] = {"is_conv": False, "Din": Din, "Cout": m.out_features}
                st = self.stats[m]
                st.A_sum += A_sum
                st.G_sum += G_sum
                st.N += N

    # --------------- public API ----------------

    def fit(self, dataloader, *, steps: int | None = None, sigma_noise: float | None = None, device: str | None = None):
        """
        Accumulate KFAC factors A,G around the current weights (MAP).
        Uses the CFM OT path to form targets.

        steps: cap number of minibatches to use (default: go through dataloader once)
        sigma_noise: override sigma for this fit (else use ctor value)
        """
        if sigma_noise is not None:
            self.sigma = float(sigma_noise)
        device = device or self.device
        self.model.to(device)

        # register hooks
        hp, handles = self._register_hooks()

        n_steps = 0
        with eval_mode(self.model):  # deterministic pass
            for batch in dataloader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x1, labels = batch
                else:
                    x1, labels = batch["images"], batch.get("labels", None)
                x1 = x1.to(device, non_blocking=True)

                B = x1.size(0)
                t = torch.rand(B, device=device)
                xt, v_star = _ot_path_targets(x1, t)

                # Gaussian negative log-likelihood on velocity
                pred = self.model(xt, t, labels.to(device) if labels is not None else None, train=False)
                resid = (pred - v_star) / (self.sigma + 1e-8)
                nll = 0.5 * (resid ** 2).mean()

                # backward to populate grad_output for modules
                self.model.zero_grad(set_to_none=True)
                nll.backward()

                # accumulate factors
                self._accumulate_batch(hp)

                # clear per-batch storage
                hp.fwd_act.clear()
                hp.bwd_grad.clear()

                n_steps += 1
                if steps is not None and n_steps >= steps:
                    break

        # remove hooks
        for h in handles:
            h.remove()

        # finalize factors (mean + damping) and cholesky of precision factors
        for m, st in self.stats.items():
            A = st.A_sum / max(1, st.N)
            G = st.G_sum / max(1, st.N)

            # damping
            dA = self.damping
            dG = self.damping
            A = A + dA * torch.eye(A.size(0), device=A.device)
            G = G + dG * torch.eye(G.size(0), device=G.device)

            # store factors
            self.A_factors[m] = A
            self.G_factors[m] = G

            # precision scale = data_size / sigma^2
            scale = (st.N / max(1.0, self.sigma ** 2))
            self.scales[m] = scale

            # Cholesky of precision factors (use sqrt scaling so Pr ⊗ Pc = scale*G⊗A)
            Pr = torch.linalg.cholesky(scale ** 0.5 * G)   # [Cout,Cout]
            Pc = torch.linalg.cholesky(scale ** 0.5 * A)   # [Din,Din]
            self.Lr[m] = Pr
            self.Lc[m] = Pc

    # --------------- sampling weights ----------------

    @torch.no_grad()
    def _sample_module_delta(self, m: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Sample a Gaussian weight/bias delta for module m from KFAC Laplace posterior.
        Returns (dW, dB) with same shapes as m.weight/m.bias.
        """
        meta = self.meta[m]
        is_conv = meta["is_conv"]
        A = self.A_factors[m]
        G = self.G_factors[m]
        Lr = self.Lr[m]   # chol of precision row factor (scaled G)
        Lc = self.Lc[m]   # chol of precision col factor (scaled A)

        Cout = meta["Cout"]
        Din = meta["Din"]    # includes +1 if bias present

        # Draw Z ~ N(0,I) of shape [Cout, Din]
        Z = torch.randn(Cout, Din, device=Lr.device, dtype=m.weight.dtype)

        # Apply row precision^{-1/2}: Y = Lr^{-1} Z
        Y = _chol_precision_solve(Lr, Z, upper=False)      # shape [Cout, Din]

        # Apply column precision^{-T}: W_delta = Y @ Lc^{-T}
        # i.e., solve (Lc^T) X^T = Y^T  -> X = (Lc^{-T} Y^T)^T
        X_t = torch.linalg.solve_triangular(Lc.T, Y.T, upper=True)  # [Din, Cout]
        W_delta_mat = X_t.T  # [Cout, Din]

        # Split weight vs bias column (if present)
        has_bias = (m.bias is not None)
        if has_bias:
            W_col = W_delta_mat[:, :-1]
            b_col = W_delta_mat[:, -1]
        else:
            W_col = W_delta_mat
            b_col = None

        # Reshape to original parameter shapes
        if is_conv:
            Cin = m.in_channels
            kH, kW = m.kernel_size
            dW = W_col.reshape(Cout, Cin, kH, kW)
        else:
            dW = W_col.reshape_as(m.weight)
        dB = b_col.reshape_as(m.bias) if (has_bias and m.bias is not None) else None
        return dW, dB

    @torch.no_grad()
    def sample_state_dict(self) -> dict:
        """
        Return a sampled state_dict (weights = MAP + sampled delta) over selected modules.
        Non-selected modules keep their MAP weights.
        """
        sd = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        for m in self.modules:
            dW, dB = self._sample_module_delta(m)
            # add deltas
            sd_id = id(m)
            # find parameter names (search once)
            # robust: scan state_dict keys for tensors that point to m.weight/m.bias
            # (simplify: assign directly)
            m.weight.data.add_(dW)  # temporarily add to live module
            if m.bias is not None and dB is not None:
                m.bias.data.add_(dB)
        # capture mutated state_dict
        sd = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        # restore model back to MAP
        self.model.load_state_dict(self._orig_state, strict=True)
        return sd

    # --------------- posterior prediction ----------------

    @torch.no_grad()
    def posterior_predict(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None,
                          n_samples: int = 1, reduce: str = "none", device: str | None = None):
        """
        Draw n_samples weight samples and compute v(x,t,y) for each.
        reduce: "none" | "mean" | "var" | "mean_var"
        """
        assert len(self.modules) > 0, "Call fit() first to build factors."
        device = device or self.device
        x = x.to(device); t = t.to(device); y = None if y is None else y.to(device)

        # keep an original copy to restore after sampling
        orig = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.model.to(device).eval()

        draws = []
        for _ in range(n_samples):
            # sample deltas and apply in-place to the live model
            for m in self.modules:
                dW, dB = self._sample_module_delta(m)
                m.weight.add_(dW)
                if m.bias is not None and dB is not None:
                    m.bias.add_(dB)

            v = self.model(x, t, y, train=False).detach().cpu()
            draws.append(v)

            # remove deltas to go back to MAP before next draw
            for m in self.modules:
                dW, dB = self._sample_module_delta(m)
                m.weight.sub_(dW)
                if m.bias is not None and dB is not None:
                    m.bias.sub_(dB)

        # robust restore (in case of drift)
        self.model.load_state_dict(orig, strict=True)

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

