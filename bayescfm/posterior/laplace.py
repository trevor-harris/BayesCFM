import torch
import torch.nn as nn
from typing import Optional, Tuple

class LaplaceCFM:
    def __init__(self, model: nn.Module, approx: str = 'last_layer', layer_pattern: str = 'last'):
        self.model = model
        self.approx = approx
        self.layer_pattern = layer_pattern
        self._fitted = False
        self._layers = self._collect_layers()

    def _collect_layers(self):
        layers = []
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                layers.append((name, m))
        return layers

    def fit(self, loader, device=None, damping: float = 1e-4):
        device = device or next(self.model.parameters()).device
        self.model.eval()
        self.damping = damping
        # Simple empirical Fisher diagonal per-parameter (fallback when KFAC unavailable)
        self.emp_fisher = {n: torch.zeros_like(p, device='cpu') for n, p in self.model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    x1 = batch[0].to(device); y = batch[1].to(device) if len(batch) > 1 else None
                elif isinstance(batch, dict):
                    x1 = batch.get('images').to(device); y = batch.get('labels', None); y = y.to(device) if y is not None else None
                else:
                    x1, y = batch.to(device), None
                B = x1.size(0)
                t = torch.rand(B, device=device)
                x1.requires_grad_(True)
                v = self.model(x1, t, y, train=False)
                # use output variance as proxy loss; accumulate grad^2
                loss = (v.detach() - v).pow(2).mean()  # zero, but keeps shapes
                g = torch.autograd.grad((v**2).mean(), [p for p in self.model.parameters() if p.requires_grad], retain_graph=False, allow_unused=True)
                for (name, p), gi in zip(self.model.named_parameters(), g):
                    if p.requires_grad and gi is not None:
                        self.emp_fisher[name] += (gi.detach().cpu()**2)
        for k in self.emp_fisher:
            self.emp_fisher[k] = self.emp_fisher[k] / max(1, len(loader))
        self._fitted = True

    @torch.no_grad()
    def sample_velocity(self, x, t, y=None, n_samples=8, reduce='mean_std'):
        assert self._fitted, "call fit() first"
        device = next(self.model.parameters()).device
        x = x.to(device); t = t.to(device); y = y.to(device) if y is not None else None
        params = {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}
        outs = []
        for _ in range(n_samples):
            # diagonal Gaussian approx
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    std = (self.emp_fisher[n].to(device) + self.damping)**-0.5
                    p.copy_(params[n] + torch.randn_like(p) * std)
            outs.append(self.model(x, t, y, train=False))
        V = torch.stack(outs, dim=0)
        if reduce == 'mean_std':
            return V.mean(0), V.var(0, unbiased=False).sqrt()
        elif reduce == 'mean_var':
            return V.mean(0), V.var(0, unbiased=False)
        else:
            return V
