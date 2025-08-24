import torch
from .core import fid_from_stats

@torch.no_grad()
def intrafid_per_class(real_feats, real_labels, fake_feats, fake_labels):
    import numpy as np
    classes = torch.unique(real_labels).tolist()
    out = {}
    for c in classes:
        r = real_feats[real_labels==c].cpu().numpy()
        f = fake_feats[fake_labels==c].cpu().numpy()
        if len(r) < 5 or len(f) < 5: continue
        mu_r, sig_r = r.mean(0), np.cov(r, rowvar=False)
        mu_f, sig_f = f.mean(0), np.cov(f, rowvar=False)
        out[int(c)] = fid_from_stats(mu_r, sig_r, mu_f, sig_f)
    return out

def rare_class_mask(labels_real, percentile=20.0):
    import numpy as np
    vals, counts = torch.unique(labels_real, return_counts=True)
    thr = np.percentile(counts.cpu().numpy(), percentile)
    rare = vals[counts.float() <= thr]
    return rare.tolist()

@torch.no_grad()
def pr_coverage_by_class(fr, yr, ff, yf, k=5, only_classes=None):
    from .prdc import prdc
    classes = only_classes if only_classes is not None else torch.unique(yr).tolist()
    out = {}
    for c in classes:
        r = fr[yr==c]
        f = ff[yf==c]
        if r.size(0) < k+2 or f.size(0) < k+2: continue
        out[int(c)] = prdc(r, f, k=k)
    return out
