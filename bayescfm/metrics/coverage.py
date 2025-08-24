import torch
from typing import Optional

@torch.no_grad()
def _lpips_or_vgg_distance(a, b, backend="lpips"):
    device = a.device
    if backend.lower() == "lpips":
        import lpips
        net = lpips.LPIPS(net='alex').to(device).eval()
        if a.min() < -1.01 or a.max() > 1.01:
            a = a.clamp(0,1)*2-1; b = b.clamp(0,1)*2-1
        return net(a,b).view(-1)
    else:
        import torchvision.models as models
        import torch.nn.functional as F
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval().to(device)
        for p in vgg.parameters(): p.requires_grad_(False)
        def prep(x):
            if x.min() < -0.01 or x.max() > 1.01: x = (x.clamp(-1,1)+1.0)*0.5
            mean = torch.tensor([0.485,0.456,0.406], device=device)[None,:,None,None]
            std  = torch.tensor([0.229,0.224,0.225], device=device)[None,:,None,None]
            return (x-mean)/std
        def feats(z):
            z = prep(z); outs=[]; cur=z
            for i,layer in enumerate(vgg):
                cur = layer(cur)
                if i in (3,8,15,22): outs.append(torch.flatten(F.adaptive_avg_pool2d(cur,(7,7)),1))
            return torch.cat(outs,1)
        return torch.norm(feats(a)-feats(b), dim=1)

@torch.no_grad()
def perceptual_coverage_k(real_images, cand_images, threshold, backend="lpips"):
    if cand_images.dim() != 5: raise ValueError("cand_images must be [N,K,C,H,W]")
    N, K = cand_images.size(0), cand_images.size(1)
    device = real_images.device
    hits = 0
    bs = 32
    for i in range(N):
        r = real_images[i].unsqueeze(0)
        best = torch.tensor(float("inf"), device=device)
        for s in range(0, K, bs):
            c = cand_images[i, s:s+bs]
            r_rep = r.expand(c.size(0), -1, -1, -1)
            d = _lpips_or_vgg_distance(r_rep, c, backend=backend)
            m = d.min()
            if m < best: best = m
        hits += int(best.item() <= threshold)
    return float(hits / N)
