
# examples/train_smoke_ad.py
import torch
from torch.utils.data import Dataset, DataLoader
import bayescfm as bcfm

class WhiteNoise(Dataset):
    def __init__(self, n=128, shape=(3,32,32), seed=0):
        g = torch.Generator().manual_seed(seed)
        C,H,W = shape
        self.x = (torch.rand(n, C, H, W, generator=g)*2-1).float()
        self.y = torch.randint(0, 10, (n,), generator=g)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class Dummy(torch.nn.Module):
        def forward(self, x, t, y=None, train=True): return torch.zeros_like(x)
    model = Dummy().to(device)
    loader = DataLoader(WhiteNoise(), batch_size=32, shuffle=True)
    ema = bcfm.train_cgm(
        model, loader, epochs=1, lr=1e-3, device=device,
        lambda_curl=1e-4, lambda_mono=1e-4,
        probes=2, probe_dist="rademacher",
        normalize_probes=True, orthogonalize=True, normalize_curl=True,
        penalty_train_flag=True,
        pool_to=(16,16), enforce_stride=False,
    )
    print("OK")
