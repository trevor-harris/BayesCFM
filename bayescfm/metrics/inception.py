import torch, torchvision
import torch.nn.functional as F
from typing import Tuple

class InceptionFeatures(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inception = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(self.device).eval()
        for p in self.inception.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.min() < -0.01 or x.max() > 1.01:
            x = (x.clamp(-1,1) + 1.0) * 0.5
        x = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
        logits = self.inception(x)
        # pool3 features proxy: use penultimate features via avgpool of Mixed_7c output
        feats = self.inception.avgpool(self.inception.Mixed_7c(self.inception.Mixed_7b(self.inception.Mixed_7a(self.inception.Mixed_7a(self.inception.Mixed_6e(self.inception.Mixed_6d(self.inception.Mixed_6c(self.inception.Mixed_6b(self.inception.Mixed_6a(self.inception.Mixed_5d(self.inception.Mixed_5c(self.inception.Mixed_5b(self.inception.Conv2d_4a_3x3(self.inception.Conv2d_3b_1x1(self.inception.Conv2d_2b_3x3(self.inception.Conv2d_2a_3x3(self.inception.Conv2d_1a_3x3(x))))))))))))))))))))
        feats = torch.flatten(feats, 1)
        probs = torch.softmax(logits, dim=1)
        return feats, probs
