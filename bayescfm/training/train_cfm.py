import torch
from ..losses.cfm import cfm_loss_ot

def train_cfm(model, loader, *, epochs=1, lr=2e-4, device='cuda', weight_decay=0.0, grad_clip=0.0, ema_decay=0.999):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = torch.optim.swa_utils.AveragedModel(model)
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
            loss = cfm_loss_ot(model, x1, t, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); ema.update_parameters(model)
    return ema
