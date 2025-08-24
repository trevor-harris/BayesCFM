import torch, torch.nn.functional as F

@torch.no_grad()
def lpips_dispersion(images, max_pairs=10000, backend="vgg"):
    N = images.size(0)
    if backend == "lpips":
        try:
            import lpips
        except Exception as e:
            raise ImportError("backend='lpips' requires the 'lpips' package") from e
        net = lpips.LPIPS(net='alex').eval().to(images.device)
        def dist(a,b): return net(a,b).view(-1)
    else:
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval().to(images.device)
        for p in vgg.parameters(): p.requires_grad_(False)
        def prep(x):
            if x.min() < -0.01 or x.max() > 1.01: x = (x.clamp(-1,1)+1.0)*0.5
            mean = torch.tensor([0.485,0.456,0.406], device=x.device)[None,:,None,None]
            std = torch.tensor([0.229,0.224,0.225], device=x.device)[None,:,None,None]
            return (x-mean)/std
        def feat(x):
            x = prep(x); cur = x; outs = []
            for i,layer in enumerate(vgg):
                cur = layer(cur)
                if i in (3,8,15,22): outs.append(F.adaptive_avg_pool2d(cur,(7,7)).flatten(1))
            return torch.cat(outs,1)
        def dist(a,b): return torch.norm(feat(a)-feat(b), dim=1)
    # sample pairs
    import numpy as np
    rng = np.random.default_rng(0)
    M = min(max_pairs, N*(N-1)//2)
    idx1 = torch.tensor(rng.integers(0,N,M), device=images.device)
    idx2 = torch.tensor(rng.integers(0,N,M), device=images.device)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    vals = []
    bs = 64
    for i in range(0, idx1.numel(), bs):
        a = images.index_select(0, idx1[i:i+bs])
        b = images.index_select(0, idx2[i:i+bs])
        vals.append(dist(a,b).detach().cpu())
    return float(torch.cat(vals).mean().item())
