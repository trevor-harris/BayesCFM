
# examples/metrics_demo.py
import torch
from torch.utils.data import Dataset, DataLoader
import bayescfm as bcfm

class WhiteNoise(Dataset):
    def __init__(self, n=256, shape=(3,32,32), seed=0):
        g = torch.Generator().manual_seed(seed)
        C,H,W = shape
        self.x = (torch.rand(n, C, H, W, generator=g)*2-1).float()
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inc = bcfm.metrics.InceptionFeatures(device=torch.device(device))

    real = WhiteNoise(n=128, seed=0)
    fake = WhiteNoise(n=128, seed=1)
    rl = DataLoader(real, batch_size=32)
    fl = DataLoader(fake, batch_size=32)

    mu_r, sig_r, _ = bcfm.metrics.compute_dataset_stats_from_loader(rl, inc)
    mu_f, sig_f, _ = bcfm.metrics.compute_dataset_stats_from_loader(fl, inc)
    fid = bcfm.metrics.fid_from_stats(mu_r, sig_r, mu_f, sig_f)
    print("FID(noise vs noise):", fid)

    probs = []
    for x in fl:
        _, p = inc(x)
        probs.append(p.cpu())
    probs = torch.cat(probs, dim=0)
    is_mean, is_std = bcfm.metrics.inception_score_from_probs(probs, splits=4)
    print("IS(noise):", is_mean, "+/-", is_std)
