from __future__ import annotations
import math
from typing import Sequence, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding for t in [0,1]. Returns [B, dim]."""
    if t.ndim == 1:
        t = t[:, None]
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device) / half)
    args = 2 * math.pi * t * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResBlock(nn.Module):
    """Residual block with FiLM conditioning: in_ch -> out_ch."""
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float = 0.0, groups: int = 32):
        super().__init__()
        self.in_ch, self.out_ch = in_ch, out_ch
        self.norm1  = nn.GroupNorm(groups, in_ch)
        self.act1   = SiLU()
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(emb_ch, 2 * out_ch)  # scale+shift

        self.norm2  = nn.GroupNorm(groups, out_ch)
        self.act2   = SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=1)  # [B, 2*out_ch]
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(h)))
        return self.skip(x) + h

class AttentionBlock(nn.Module):
    """2D multi-head self attention with conv QKV, conv proj."""
    def __init__(self, channels: int, num_heads: int = 4, groups: int = 32):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)  # [B,Hd,Dim]->[B,heads,HW,dim]
        k = self.k(h_).view(b, self.num_heads, self.head_dim, h * w)                   # [B,heads,dim,HW]
        v = self.v(h_).view(b, self.num_heads, self.head_dim, h * w).transpose(2, 3)  # [B,heads,HW,dim]
        attn = torch.matmul(q, k) / math.sqrt(self.head_dim)                           # [B,heads,HW,HW]
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                                                    # [B,heads,HW,dim]
        out = out.transpose(2, 3).contiguous().view(b, c, h, w)                        # merge heads
        out = self.proj(out)
        return x + out

class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class UNetCFM(nn.Module):
    """
    U-Net backbone for Conditional Flow Matching.

    attn_resolutions: spatial sizes (e.g., (32,16)) where attention is applied.
                      We add attention after each resblock whose current min(H,W) hits
                      one of these values (and also at the bottleneck).
    """
    def __init__(
        self,
        *,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 3,
        channel_mult: Sequence[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        num_heads: int = 4,
        dropout: float = 0.0,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.10,
        time_embed_mult: int = 4,
        groupnorm_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.channel_mult = tuple(channel_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.groupnorm_groups = groupnorm_groups

        # time/class embedding
        time_dim = model_channels * time_embed_mult
        self.time_fc1 = nn.Linear(time_dim, time_dim)
        self.time_fc2 = nn.Linear(time_dim, time_dim)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes + 1, time_dim)  # +1 for NULL
            self.null_class_id = num_classes
        else:
            self.label_emb = None
            self.null_class_id = None

        # stem
        ch = model_channels
        self.stem = nn.Conv2d(in_channels, ch, 3, padding=1)

        # encoder
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.enc_attn:   nn.ModuleList = nn.ModuleList()  # mirror enc_blocks; each is a ModuleList of attn blocks
        self.downsamples: nn.ModuleList = nn.ModuleList()
        input_ch = ch
        self.enc_channel_history: List[int] = [ch]  # for decoder skips

        for i, mult in enumerate(self.channel_mult):
            out_ch = model_channels * mult
            stage_blocks = nn.ModuleList()
            stage_attn   = nn.ModuleList()

            # first block (in_ch -> out_ch)
            stage_blocks.append(ResBlock(input_ch, out_ch, emb_ch=time_dim, dropout=dropout, groups=groupnorm_groups))
            stage_attn.append(AttentionBlock(out_ch, num_heads=num_heads, groups=groupnorm_groups))
            input_ch = out_ch

            # remaining blocks (out_ch -> out_ch)
            for _ in range(self.num_res_blocks - 1):
                stage_blocks.append(ResBlock(out_ch, out_ch, emb_ch=time_dim, dropout=dropout, groups=groupnorm_groups))
                stage_attn.append(AttentionBlock(out_ch, num_heads=num_heads, groups=groupnorm_groups))

            self.enc_blocks.append(stage_blocks)
            self.enc_attn.append(stage_attn)

            # record skips: one per resblock
            self.enc_channel_history.extend([out_ch] * self.num_res_blocks)

            if i != len(self.channel_mult) - 1:
                self.downsamples.append(Downsample(out_ch, out_ch))
                self.enc_channel_history.append(out_ch)  # skip after downsample

        # bottleneck
        self.mid_block1 = ResBlock(input_ch, input_ch, emb_ch=time_dim, dropout=dropout, groups=groupnorm_groups)
        self.mid_attn   = AttentionBlock(input_ch, num_heads=num_heads, groups=groupnorm_groups)
        self.mid_block2 = ResBlock(input_ch, input_ch, emb_ch=time_dim, dropout=dropout, groups=groupnorm_groups)

        # decoder
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.dec_attn:   nn.ModuleList = nn.ModuleList()
        self.upsamples:  nn.ModuleList = nn.ModuleList()
        dec_in_ch = input_ch
        skip_iter = iter(reversed(self.enc_channel_history))
        for i, mult in reversed(list(enumerate(self.channel_mult))):
            out_ch = model_channels * mult
            stage_blocks = nn.ModuleList()
            stage_attn   = nn.ModuleList()
            # num_res_blocks + 1 to consume the skip after downsample as well
            for _ in range(self.num_res_blocks + 1):
                skip_ch = next(skip_iter)
                stage_blocks.append(ResBlock(dec_in_ch + skip_ch, out_ch, emb_ch=time_dim, dropout=dropout, groups=groupnorm_groups))
                stage_attn.append(AttentionBlock(out_ch, num_heads=num_heads, groups=groupnorm_groups))
                dec_in_ch = out_ch
            self.dec_blocks.append(stage_blocks)
            self.dec_attn.append(stage_attn)
            if i != 0:
                self.upsamples.append(Upsample(dec_in_ch, out_ch))
                dec_in_ch = out_ch

        # head
        self.head_norm = nn.GroupNorm(groupnorm_groups, dec_in_ch)
        self.head_act  = SiLU()
        self.head = nn.Conv2d(dec_in_ch, out_channels, 3, padding=1)

    # -- embedding --------------------------------------------------------------

    def _time_label_embed(self, t: torch.Tensor, y: Optional[torch.Tensor], train: bool) -> torch.Tensor:
        time_dim = self.time_fc1.in_features
        te = timestep_embedding(t, time_dim)
        te = self.time_fc2(F.silu(self.time_fc1(te)))
        if self.label_emb is None or y is None:
            return te
        if train and self.class_dropout_prob > 0:
            keep = torch.rand_like(y.float()) > self.class_dropout_prob
            y = torch.where(keep.bool(), y, torch.full_like(y, self.null_class_id))
        ye = self.label_emb(y)
        return F.silu(te + ye)

    # -- forward ---------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None, *, train: bool = True) -> torch.Tensor:
        """
        x: [B,C,H,W] in [-1,1]; t: [B] in [0,1]; y: [B] ints (optional)
        returns: velocity field [B,out_channels,H,W]
        """
        B, C, H, W = x.shape
        emb = self._time_label_embed(t, y, train=train)

        skips: List[torch.Tensor] = []

        # ----- encoder -----
        cur = self.stem(x)
        skips.append(cur)
        for i, (stage_blocks, stage_attn) in enumerate(zip(self.enc_blocks, self.enc_attn)):
            for block, attn in zip(stage_blocks, stage_attn):
                cur = block(cur, emb)
                # apply attention at this resolution if requested
                if min(cur.shape[2], cur.shape[3]) in self.attn_resolutions:
                    cur = attn(cur)
                skips.append(cur)
            # downsample between stages (use index instead of .index())
            if i != len(self.enc_blocks) - 1:
                cur = self.downsamples[i](cur)
                skips.append(cur)

        # ----- bottleneck -----
        cur = self.mid_block1(cur, emb)
        cur = self.mid_attn(cur)
        cur = self.mid_block2(cur, emb)

        # ----- decoder -----
        for i, (stage_blocks, stage_attn) in enumerate(zip(self.dec_blocks, self.dec_attn)):
            for block, attn in zip(stage_blocks, stage_attn):
                skip = skips.pop()
                cur = torch.cat([cur, skip], dim=1)
                cur = block(cur, emb)
                if min(cur.shape[2], cur.shape[3]) in self.attn_resolutions:
                    cur = attn(cur)
            if i != len(self.dec_blocks) - 1:
                cur = self.upsamples[i](cur)

        # ----- head -----
        cur = self.head_act(self.head_norm(cur))
        v = self.head(cur)
        return v

