import torch
from torch.utils.data import Dataset, DataLoader
import bayescfm as bcfm

class WhiteNoise(Dataset):
    def __init__(self, n=256, shape=(3,32,32), seed=0):
        g = torch.Generator().manual_seed(seed)
        C,H,W = shape
        self.x = (torch.rand(n, C, H, W, generator=g)*2-1).float()
        self.y = torch.randint(0, 10, (n,), generator=g)
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = bcfm.UNetCFM(in_channels=3, out_channels=3, model_channels=64, channel_mult=(1,2,2), num_res_blocks=1, attn_resolutions=(16,), num_heads=4, num_classes=10)
loader = DataLoader(WhiteNoise(), batch_size=64, shuffle=True)
ema = bcfm.train_cfm(model, loader, epochs=1, lr=2e-4, device=device)
print("OK")
