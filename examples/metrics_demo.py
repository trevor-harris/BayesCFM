import torch, bayescfm as bcfm
from torch.utils.data import Dataset, DataLoader

class Noise(Dataset):
    def __init__(self, n=200, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = (torch.rand(n,3,32,32,generator=g)*2-1).float()
        self.y = torch.randint(0,10,(n,),generator=g)
    def __len__(self): return self.x.size(0)
    def __getitem__(self,i): return self.x[i], self.y[i]

real = DataLoader(Noise(200,0), batch_size=64)
fake = DataLoader(Noise(200,1), batch_size=64)

inc = bcfm.metrics.InceptionFeatures()
mu_r, sig_r, _ = bcfm.metrics.compute_dataset_stats_from_loader(real, inc)
mu_f, sig_f, _ = bcfm.metrics.compute_dataset_stats_from_loader(fake, inc)
print("FID:", bcfm.metrics.fid_from_stats(mu_r, sig_r, mu_f, sig_f))
