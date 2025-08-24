
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision.models import Inception_V3_Weights
from typing import Tuple, Optional

class InceptionFeatures(nn.Module):
    """Extract pool3 features (2048) and softmax probs for IS."""
    def __init__(
        self,
        device: Optional[torch.device] = None,
        weights: Optional[Inception_V3_Weights] = Inception_V3_Weights.IMAGENET1K_V1,
        resize_input: bool = True,
        input_range: str = "[-1,1]",
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = inception_v3(weights=weights, transform_input=False, aux_logits=False)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.resize_input = resize_input
        assert input_range in ("[-1,1]", "[0,1]")
        self.input_range = input_range

    @torch.no_grad()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_range == "[-1,1]":
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)
        elif x.shape[1] > 3:
            x = x[:, :3]
        if self.resize_input:
            x = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None,:,None,None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None,:,None,None]
        x = (x - mean) / std
        return x

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.preprocess(x.to(self.device, non_blocking=True))
        m = self.model
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1,1))
        feats = torch.flatten(x, 1)
        logits = m.fc(torch.nn.functional.dropout(feats, training=False))
        probs = torch.softmax(logits, dim=1)
        return feats, probs

