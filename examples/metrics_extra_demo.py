
# examples/metrics_extra_demo.py
import torch
from torch.utils.data import Dataset, DataLoader
import bayescfm as bcfm

class LabeledNoise(Dataset):
    def __init__(self, n=300, shape=(3,32,32), num_classes=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        C,H,W = shape
        self.x = (torch.rand(n, C, H, W, generator=g)*2-1).float()
        self.y = torch.randint(0, num_classes, (n,), generator=g)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inc = bcfm.metrics.InceptionFeatures(device=torch.device(device))

    real = LabeledNoise(n=200, seed=0)
    fake = LabeledNoise(n=200, seed=1)
    rl = DataLoader(real, batch_size=64)
    fl = DataLoader(fake, batch_size=64)

    # features
    def get_feats_labels(loader):
        feats, labels = [], []
        for xb, yb in loader:
            f,_ = inc(xb)
            feats.append(f.cpu())
            labels.append(yb.cpu())
        return torch.cat(feats,0), torch.cat(labels,0)
    fr, yr = get_feats_labels(rl)
    ff, yf = get_feats_labels(fl)

    # cMS-SSIM within fake by class
    X_fake = torch.stack([fake[i][0] for i in range(len(fake))], dim=0)
    y_fake = torch.stack([fake[i][1] for i in range(len(fake))], dim=0)
    try:
        cms = bcfm.metrics.cms_ssim_by_class(X_fake, y_fake, pairs_per_class=200)
        print("cMS-SSIM (fake):", {int(k): round(v,4) for k,v in cms.items()})
    except Exception as e:
        print("cMS-SSIM unavailable:", e)

    # Perceptual coverage: fabricate K candidates per target by adding noise
    N = len(real)
    K = 4
    X_real = torch.stack([real[i][0] for i in range(N)], dim=0)
    cands = X_real.unsqueeze(1) + 0.1*torch.randn(N, K, *X_real.shape[1:])
    cov = bcfm.metrics.perceptual_coverage_k(X_real, cands, threshold=0.5, backend="vgg")
    print("Perceptual coverage (K=4, thr=0.5):", cov)

    # PRDC frontier
    frontier = bcfm.metrics.prdc_frontier(fr, ff, ks=[1,3,5,10,20])
    print("PRDC frontier:", frontier)

    # MIA AUC: treat first half of 'real' as members, second half as non-members; use fake as reference set
    members = fr[:100]; nonmembers = fr[100:200]
    auc, scores = bcfm.metrics.mia_auc_nn_distance(members, nonmembers, reference_feats=ff)
    print("MIA AUC (NN distance to fake):", auc)

