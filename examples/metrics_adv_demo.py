
# examples/metrics_adv_demo.py
import torch
from torch.utils.data import Dataset, DataLoader
import bayescfm as bcfm

class LabeledNoise(Dataset):
    def __init__(self, n=512, shape=(3,32,32), num_classes=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        C,H,W = shape
        self.x = (torch.rand(n, C, H, W, generator=g)*2-1).float()
        self.y = torch.randint(0, num_classes, (n,), generator=g)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inc = bcfm.metrics.InceptionFeatures(device=torch.device(device))

    # real/fake labeled noise
    real = LabeledNoise(n=256, seed=0)
    fake = LabeledNoise(n=256, seed=1)
    rl = DataLoader(real, batch_size=64)
    fl = DataLoader(fake, batch_size=64)

    # Extract features + labels
    def get_feats_labels(loader):
        feats, labels = [], []
        for xb, yb in loader:
            f,_ = inc(xb)
            feats.append(f.cpu())
            labels.append(yb.cpu())
        return torch.cat(feats,0), torch.cat(labels,0)

    fr, yr = get_feats_labels(rl)
    ff, yf = get_feats_labels(fl)

    # PRDC
    print("PRDC:", bcfm.metrics.prdc(fr, ff, k=5))

    # Classwise Intra-FID
    print("Intra-FID per class:", bcfm.metrics.intrafid_per_class(fr, yr, ff, yf))

    # Rare-class mask (bottom 20% by freq) then PR/Coverage on those
    rare = bcfm.metrics.rare_class_mask(yr, percentile=20.0)
    print("rare classes:", rare.tolist())
    print("PR/Coverage (rare):", bcfm.metrics.pr_coverage_by_class(fr, yr, ff, yf, k=5, only_classes=rare))

    # Duplicate rate (feature space) within fake set & vs real
    print("dup rate (fake):", bcfm.metrics.duplicate_rate(ff, threshold=0.1))
    print("dup rate (fake vs real):", bcfm.metrics.duplicate_rate_vs_ref(ff, fr, threshold=0.1))

    # Birthday test
    print("birthday:", bcfm.metrics.birthday_paradox_test(ff, subset=128, threshold=0.1, trials=3))

    # LPIPS dispersion (will require 'lpips' if backend='lpips')
    x_fake = torch.stack([fake[i][0] for i in range(len(fake))], dim=0)
    try:
        disp = bcfm.metrics.lpips_dispersion(x_fake, max_pairs=500, backend="vgg", device=torch.device(device))
        print("LPIPS dispersion (vgg-proxy):", disp)
    except Exception as e:
        print("LPIPS dispersion not available:", e)

