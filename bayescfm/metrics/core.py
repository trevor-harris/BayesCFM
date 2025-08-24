import torch, numpy as np
from typing import Tuple

def compute_activations_from_images(images, extractor, batch_size=64):
    feats, probs = [], []
    for i in range(0, images.size(0), batch_size):
        f, p = extractor(images[i:i+batch_size])
        feats.append(f.cpu()); probs.append(p.cpu())
    return torch.cat(feats,0), torch.cat(probs,0)

def compute_dataset_stats_from_loader(loader, extractor):
    feats = []
    for xb, *_ in loader:
        f,_ = extractor(xb)
        feats.append(f.cpu())
    feats = torch.cat(feats,0).numpy()
    mu = feats.mean(0); sig = np.cov(feats, rowvar=False)
    return mu, sig, feats

def fid_from_stats(mu_r, sig_r, mu_f, sig_f, eps=1e-6):
    import numpy as np
    from scipy.linalg import sqrtm
    covmean = sqrtm((sig_r @ sig_f).astype(np.float64))
    if np.iscomplexobj(covmean): covmean = covmean.real
    diff = mu_r - mu_f
    return float(diff.dot(diff) + np.trace(sig_r + sig_f - 2*covmean))

def kid_from_activations(fr, ff, subset_size=1000, n_subsets=100, degree=3, gamma=None, coef0=1.0):
    # polynomial MMD^2 unbiased estimate
    import numpy as np
    if gamma is None: gamma = 1.0 / fr.shape[1]
    def poly(x, y): return (gamma * (x @ y.T) + coef0) ** degree
    m = min(subset_size, fr.shape[0], ff.shape[0])
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(n_subsets):
        i = rng.choice(fr.shape[0], m, replace=False)
        j = rng.choice(ff.shape[0], m, replace=False)
        X, Y = fr[i], ff[j]
        k_xx = poly(X, X); k_yy = poly(Y, Y); k_xy = poly(X, Y)
        # unbiased MMD^2
        mmd2 = (k_xx.sum()-np.trace(k_xx)) / (m*(m-1)) + (k_yy.sum()-np.trace(k_yy)) / (m*(m-1)) - 2*k_xy.mean()
        scores.append(mmd2)
    return float(np.mean(scores)), float(np.std(scores)/np.sqrt(n_subsets))

def inception_score_from_probs(probs, splits=10):
    import numpy as np
    N = probs.shape[0]
    scores = []
    for i in range(splits):
        part = probs[i * (N // splits): (i+1) * (N // splits), :]
        py = part.mean(0, keepdims=True)
        kl = (part * (np.log(part + 1e-12) - np.log(py + 1e-12))).sum(1)
        scores.append(np.exp(kl.mean()))
    return float(np.mean(scores)), float(np.std(scores))
