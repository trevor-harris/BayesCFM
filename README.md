# bayes-cfm

Bayesian Conditional Flow Matching (CFM) in PyTorch — UNet backbone, CFM training, **autodiff-based** Jacobian (Hutchinson) regularizers, SG-MCMC + Laplace posteriors, ODE samplers, and a comprehensive metrics suite.

## Install
```bash
pip install -e .
pip install -e .[metrics]   # optional extras for LPIPS, MS-SSIM
```

## Quickstart
```python
import torch
from torch.utils.data import DataLoader, Dataset
import bayescfm as bcfm

model = bcfm.UNetCFM(in_channels=3, out_channels=3, model_channels=64,
                     channel_mult=(1,2,2), num_res_blocks=1,
                     attn_resolutions=(16,), num_heads=4,
                     num_classes=10)

class WhiteNoise(Dataset):
    def __init__(self, n=512, shape=(3,32,32), seed=0):
        g = torch.Generator().manual_seed(seed)
        C,H,W = shape
        self.x = (torch.rand(n, C, H, W, generator=g)*2-1).float()
        self.y = torch.randint(0, 10, (n,), generator=g)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]

loader = DataLoader(WhiteNoise(), batch_size=64, shuffle=True)
ema = bcfm.train_cfm(model, loader, epochs=1, lr=2e-4, device="cuda" if torch.cuda.is_available() else "cpu")
```

### Regularized training (autodiff Jacobian penalties)
```python
ema_reg = bcfm.train_cgm(
    model, loader, epochs=1, lr=2e-4, device="cuda" if torch.cuda.is_available() else "cpu",
    lambda_curl=1e-4, lambda_mono=1e-4,
    probes=2, probe_dist="rademacher",
    normalize_probes=True, orthogonalize=True, normalize_curl=True,
    penalty_train_flag=True,
    pool_to=(64,64), enforce_stride=True,
)
```

### SG-MCMC Bayes + posterior sampling
```python
ema_bayes, posterior = bcfm.train_cgm_bayes(
    model, loader, epochs=1, lr=2e-4, device="cuda" if torch.cuda.is_available() else "cpu",
    lambda_curl=1e-4, lambda_mono=1e-4,
    probes=1, probe_dist="rademacher",
    sgmcmc_enable=True, sgmcmc_alg="sghmc", sgmcmc_eta=2e-5, sgmcmc_friction=0.05,
    sgmcmc_collect=True, sgmcmc_burnin_steps=10, sgmcmc_thin=5, sgmcmc_max_samples=10
)

x = torch.randn(4,3,32,32)
t = torch.rand(4)
y = torch.randint(0,10,(4,))
v_mean, v_var = bcfm.posterior_sampler(ema_bayes, posterior, x, t, y, n_samples=5, reduce="mean_var")
```

### Laplace approximation
```python
lap = bcfm.LaplaceCFM(model, approx="last_layer")
lap.fit(loader, device="cuda" if torch.cuda.is_available() else "cpu")
v_mean, v_std = lap.sample_velocity(x, t, y, n_samples=8, reduce="mean_std")
```

### Metrics
```python
inc = bcfm.metrics.InceptionFeatures()
# FID
mu_r, sig_r, _ = bcfm.metrics.compute_dataset_stats_from_loader(real_loader, inc)
mu_f, sig_f, _ = bcfm.metrics.compute_dataset_stats_from_loader(fake_loader, inc)
fid = bcfm.metrics.fid_from_stats(mu_r, sig_r, mu_f, sig_f)
# PRDC frontier
fr = inc(real_images)[0]; ff = inc(fake_images)[0]
front = bcfm.metrics.prdc_frontier(fr, ff, ks=[1,3,5,10,20])
# cMS-SSIM and K-hit
cms = bcfm.metrics.cms_ssim_by_class(fake_images, fake_labels, pairs_per_class=500)
k_hit = bcfm.metrics.perceptual_coverage_k(real_images, cand_images, threshold=0.5, backend="vgg")
```
