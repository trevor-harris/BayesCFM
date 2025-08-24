import torch
from ..losses.cfm import cfm_loss_ot
from ..model.unet import UNetCFM

def train_cfm(
    model: UNetCFM,
    dataloader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    device: str = "cuda",
    ema_decay: float = 0.999,
    max_grad_norm: Optional[float] = 1.0,
    log_every: int = 100,
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
            x1 = x1.to(device)
            if labels is not None:
                labels = labels.to(device)

            B = x1.shape[0]
            t = torch.rand(B, device=device)

            loss = cfm_loss_ot(model, x1, t, labels)
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
                print(f"[epoch {epoch}] step {step:>7d}  loss={loss.item():.4f}")

    # return EMA-cloned model for eval
    ema_model = UNetCFM(**{k: getattr(model, k) for k in [
        "in_channels","model_channels","out_channels","channel_mult","num_res_blocks",
        "attn_resolutions","num_heads","dropout","num_classes","class_dropout_prob",
        "groupnorm_groups"
    ]})
    ema_model.load_state_dict({k: v for (k,_), v in zip(model.state_dict().items(), ema_params)})
    ema_model.to(device).eval()
    return ema_model

