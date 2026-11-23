"""A standard U-Net (Ronneberger et al., 2015) with batch norm and bilinear upsampling, a pretrained-encoder U-Net
(segmentation_models_pytorch), or a TerraMind foundation-model backbone (terratorch; separate env, see envs/)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels: int, base: int = 32, depth: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.stem = DoubleConv(in_channels, widths[0])
        self.down = nn.ModuleList(DoubleConv(widths[i], widths[i + 1]) for i in range(depth))
        self.up = nn.ModuleList(DoubleConv(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth)))
        self.drop = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.stem(x)]
        for block in self.down:
            skips.append(block(F.max_pool2d(skips[-1], 2)))
        x = self.drop(skips.pop())
        for block in self.up:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)  # logits, shape (B, 1, H, W)


# TerraMind pre-training standardisation of S1GRD backscatter in dB (terratorch terramind_register.py, v1)
TERRAMIND_STATS = {"VV": (-12.599, 5.195), "VH": (-20.293, 5.890)}


class TerraMindSeg(nn.Module):
    """TerraMind-1.0 (Jakubik et al. 2025) S1GRD encoder + UNet decoder, as in IBM's Sen1Floods11 fine-tuning config
    (configs/terramind_v1_base_sen1floods11.yaml in IBM/terramind: layers 2/5/8/11, decoder 512-256-128-64).

    `bands` names the input channels in order. VV and VH use the pretrained S1GRD patch weights and are mapped from
    floodseg's fixed dB scaling back to dB and standardised with TerraMind's statistics; any other band (e.g. HAND,
    already scaled by floodseg.data.normalize) is an extra S1GRD channel whose patch weights start from random init.
    """

    def __init__(self, bands: list[str], size: str = "base", pretrained: bool = True) -> None:
        super().__init__()
        from terratorch.models import EncoderDecoderFactory

        self.net = EncoderDecoderFactory().build_model(
            task="segmentation", backbone=f"terramind_v1_{size}", backbone_pretrained=pretrained,
            backbone_modalities=["S1GRD"], backbone_bands={"S1GRD": list(bands)},
            necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                   {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                   {"name": "LearnedInterpolateToPyramidal"}],
            decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], head_dropout=0.1, num_classes=1)
        # floodseg vv/vh = (dB + 12.5) / 7.5  ->  (dB - mean) / std = x * 7.5 / std + (-12.5 - mean) / std
        scale = [7.5 / TERRAMIND_STATS[b][1] if b in TERRAMIND_STATS else 1.0 for b in bands]
        shift = [(-12.5 - TERRAMIND_STATS[b][0]) / TERRAMIND_STATS[b][1] if b in TERRAMIND_STATS else 0.0 for b in bands]
        self.register_buffer("scale", torch.tensor(scale).view(1, -1, 1, 1))
        self.register_buffer("shift", torch.tensor(shift).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x * self.scale + self.shift).output
        return out.unsqueeze(1) if out.dim() == 3 else out  # logits, shape (B, 1, H, W)


def build_model(in_channels: int, cfg: dict) -> nn.Module:
    """`model.arch: terramind` -> TerraMindSeg (`model.bands` names the channels); `model.encoder` set (e.g. resnet34)
    -> pretrained-encoder U-Net from segmentation_models_pytorch; else the plain UNet."""
    cfg = dict(cfg)
    if cfg.pop("arch", "unet") == "terramind":
        if len(cfg["bands"]) != in_channels:
            raise ValueError(f"model.bands {cfg['bands']} does not match {in_channels} input channels")
        return TerraMindSeg(cfg["bands"], cfg.get("size", "base"), cfg.get("pretrained", True))
    encoder = cfg.pop("encoder", None)
    if encoder is None:
        return UNet(in_channels, **cfg)
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder, encoder_weights=cfg.get("encoder_weights", "imagenet"), in_channels=in_channels, classes=1)
