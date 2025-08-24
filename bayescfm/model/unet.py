import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class SiLU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim, dropout=0.0):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.act = SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Sequential(SiLU(), nn.Linear(temb_dim, out_ch))
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity())

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act(self.norm2(h))))
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(ch, ch*3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.num_heads, C // self.num_heads, H*W)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2]  # [B,heads,C_head,HW]
        q = q.permute(0,1,3,2)  # [B,heads,HW,C_head]
        k = k.permute(0,1,2,3)  # [B,heads,C_head,HW]
        attn = torch.matmul(q, k) / math.sqrt(k.size(2))  # [B,heads,HW,HW]
        attn = torch.softmax(attn, dim=-1)
        v = v.permute(0,1,3,2)  # [B,heads,HW,C_head]
        h = torch.matmul(attn, v)  # [B,heads,HW,C_head]
        h = h.permute(0,1,3,2).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h

class Downsample(nn.Module):
    def __init__(self, ch): super().__init__(); self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x): return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch): super().__init__(); self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.op(x)

class UNetCFM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, model_channels=64,
                 channel_mult=(1,2,2), num_res_blocks=2,
                 attn_resolutions=(16,), num_heads=4,
                 num_classes=None, class_dropout_prob=0.1, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.attn_resolutions = set(attn_resolutions)
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, 4*model_channels), SiLU(),
            nn.Linear(4*model_channels, 4*model_channels),
        )
        self.temb_dim = 4*model_channels

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.temb_dim)
        else:
            self.label_emb = None

        chs = [model_channels * m for m in channel_mult]
        self.input = nn.Conv2d(in_channels, chs[0], 3, padding=1)

        # down
        downs = nn.ModuleList()
        curr = chs[0]
        self.down_res = []
        for i, m in enumerate(channel_mult):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(curr, chs[i], self.temb_dim, dropout))
                curr = chs[i]
                blocks.append(AttentionBlock(curr, num_heads) if 2**(i+1) * 8 in self.attn_resolutions else nn.Identity())
            down = nn.Module()
            down.blocks = blocks
            down.down = (Downsample(curr) if i != len(channel_mult)-1 else nn.Identity())
            downs.append(down)
        self.downs = downs

        self.mid = nn.ModuleList([
            ResBlock(curr, curr, self.temb_dim, dropout),
            AttentionBlock(curr, num_heads),
            ResBlock(curr, curr, self.temb_dim, dropout),
        ])

        # up
        ups = nn.ModuleList()
        for i in reversed(range(len(channel_mult))):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks+1):  # + skip
                blocks.append(ResBlock(curr + chs[i], chs[i], self.temb_dim, dropout))
                curr = chs[i]
                blocks.append(AttentionBlock(curr, num_heads) if 2**(i+1) * 8 in self.attn_resolutions else nn.Identity())
            up = nn.Module()
            up.blocks = blocks
            up.up = (Upsample(curr) if i != 0 else nn.Identity())
            ups.append(up)
        self.ups = ups

        self.out = nn.Sequential(
            nn.GroupNorm(32, curr), SiLU(), nn.Conv2d(curr, out_channels, 3, padding=1)
        )

    def forward(self, x, t, y=None, train=True):
        # Embed time and (optional) class label into a single temb
        t_emb = timestep_embedding(t, self.model_channels)
        temb = self.time_mlp(t_emb)
        if self.label_emb is not None and y is not None:
            if self.training and train and self.class_dropout_prob > 0.0:
                mask = (torch.rand_like(y.float()) < self.class_dropout_prob)
                y = y.masked_fill(mask, 0)  # optional: could use learned null token
            temb = temb + self.label_emb(y)

        hs = []
        h = self.input(x)
        for d in self.downs:
            for block in d.blocks:
                if isinstance(block, ResBlock):
                    h = block(h, temb)
                else:
                    h = block(h)
            hs.append(h)
            h = d.down(h)

        for block in self.mid:
            if isinstance(block, ResBlock):
                h = block(h, temb)
            else:
                h = block(h)

        for i, u in enumerate(self.ups):
            h = torch.cat([h, hs.pop()], dim=1)
            for block in u.blocks:
                if isinstance(block, ResBlock):
                    h = block(h, temb)
                else:
                    h = block(h)
            h = u.up(h)

        return self.out(h)
