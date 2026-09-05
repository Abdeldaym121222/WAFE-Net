"""
WAFE-Net: Wavelet-Attention Feature Enhancement Network
========================================================
Architecture as described in the research paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ─────────────────────────────────────────────
# 1. Dual Residual Attention Block (DRAB)
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.gap(x))
        mx  = self.mlp(self.gmp(x))
        Mc  = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return Mc * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        Ms = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return Ms * x


class ResidualAttentionUnit(nn.Module):
    """One residual unit with CBAM-style channel + spatial attention."""
    def __init__(self, channels, dropout_p=0.4):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.channel_attn  = ChannelAttention(channels)
        self.spatial_attn  = SpatialAttention()
        self.dropout        = nn.Dropout2d(p=dropout_p)
        self.relu           = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out = out + residual                # residual connection (Eq. 3)
        out = self.channel_attn(out)        # Eq. 4-5
        out = self.spatial_attn(out)        # Eq. 6-7
        out = self.dropout(out)
        return self.relu(out)


class DRAB(nn.Module):
    """
    Dual Residual Attention Block.
    Projects 1536 → 512 then applies two stacked ResidualAttentionUnits.
    """
    def __init__(self, in_channels=1536, mid_channels=512, dropout_p=0.4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.unit1 = ResidualAttentionUnit(mid_channels, dropout_p)
        self.unit2 = ResidualAttentionUnit(mid_channels, dropout_p)

    def forward(self, x):
        x = self.proj(x)
        x = self.unit1(x)
        x = self.unit2(x)
        return x  # shape: (B, 512, 7, 7)


# ─────────────────────────────────────────────
# 2. Wavelet Transform Module (WTM)
# ─────────────────────────────────────────────
class WaveletTransformModule(nn.Module):
    """
    2-D Haar DWT decomposition of feature maps into 4 subbands,
    followed by per-subband 1×1 Conv to 128 channels, then concat → 512.
    Output is upsampled to (7, 7).
    """
    def __init__(self, in_channels=1536, out_channels_per_band=128, target_size=(7, 7), dropout_p=0.3):
        super().__init__()
        self.target_size = target_size

        # 1×1 conv for each of the 4 subbands
        self.conv_ll = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_band, 1, bias=False),
            nn.BatchNorm2d(out_channels_per_band), nn.ReLU(inplace=True),
        )
        self.conv_lh = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_band, 1, bias=False),
            nn.BatchNorm2d(out_channels_per_band), nn.ReLU(inplace=True),
        )
        self.conv_hl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_band, 1, bias=False),
            nn.BatchNorm2d(out_channels_per_band), nn.ReLU(inplace=True),
        )
        self.conv_hh = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_per_band, 1, bias=False),
            nn.BatchNorm2d(out_channels_per_band), nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout_p)

    @staticmethod
    def haar_dwt_2d(x):
        """
        Apply 2D Haar DWT to feature map x ∈ (B, C, H, W).
        Returns four subbands each of shape (B, C, ceil(H/2), ceil(W/2)).
        Pads odd spatial dimensions by 1 before downsampling.
        Uses Eq. 8-11 from the paper.
        """
        # Pad to even dimensions if necessary
        _, _, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Row-wise (horizontal) filtering + downsample
        x_low  = (x[:, :, :, 0::2] + x[:, :, :, 1::2]) / 2.0   # h_L row
        x_high = (x[:, :, :, 0::2] - x[:, :, :, 1::2]) / 2.0   # h_H row

        # Column-wise (vertical) filtering + downsample
        ll = (x_low[:, :, 0::2, :]  + x_low[:, :, 1::2, :])  / 2.0
        lh = (x_low[:, :, 0::2, :]  - x_low[:, :, 1::2, :])  / 2.0
        hl = (x_high[:, :, 0::2, :] + x_high[:, :, 1::2, :]) / 2.0
        hh = (x_high[:, :, 0::2, :] - x_high[:, :, 1::2, :]) / 2.0
        return ll, lh, hl, hh

    def forward(self, x):
        ll, lh, hl, hh = self.haar_dwt_2d(x)                    # each: (B, C, H/2, W/2)

        f_ll = self.conv_ll(ll)
        f_lh = self.conv_lh(lh)
        f_hl = self.conv_hl(hl)
        f_hh = self.conv_hh(hh)

        f_w = torch.cat([f_ll, f_lh, f_hl, f_hh], dim=1)        # (B, 512, H/2, W/2)  Eq. 12
        f_w = self.dropout(f_w)

        # Upsample to match DRAB output resolution  Eq. 13
        f_w = F.interpolate(f_w, size=self.target_size, mode='bilinear', align_corners=False)
        return f_w  # (B, 512, 7, 7)


# ─────────────────────────────────────────────
# 3. Cross-Domain Fusion Module (CDF)
# ─────────────────────────────────────────────
class CrossDomainFusion(nn.Module):
    """
    Adaptive fusion of spatial (y'') and frequency (F'_w) features.
    Implements Eq. 14-18.
    """
    def __init__(self, channels=512):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Learnable scalar attention weights (Eq. 16)
        self.w_s = nn.Linear(channels, 1, bias=False)
        self.w_w = nn.Linear(channels, 1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, y_pp, f_w_prime):
        # Eq. 14-15: concatenate and project
        f_cat   = torch.cat([y_pp, f_w_prime], dim=1)           # (B, 1024, 7, 7)
        f_fused = self.conv_fuse(f_cat)                          # (B, 512,  7, 7)

        # Eq. 16: scalar attention weights from GAP embeddings
        gap_s = self.gap(y_pp).flatten(1)                        # (B, 512)
        gap_w = self.gap(f_w_prime).flatten(1)                   # (B, 512)
        alpha_s = torch.sigmoid(self.w_s(gap_s))                 # (B, 1)
        alpha_w = torch.sigmoid(self.w_w(gap_w))                 # (B, 1)

        alpha_s = alpha_s.unsqueeze(-1).unsqueeze(-1)            # (B,1,1,1)
        alpha_w = alpha_w.unsqueeze(-1).unsqueeze(-1)

        # Eq. 17
        f_out = alpha_s * y_pp + alpha_w * f_w_prime + f_fused   # (B, 512, 7, 7)
        return f_out


# ─────────────────────────────────────────────
# 4. Full WAFE-Net
# ─────────────────────────────────────────────
class WAFENet(nn.Module):
    """
    WAFE-Net: Wavelet-Attention Feature Enhancement Network
    -------------------------------------------------------
    num_classes : 7  (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    """
    def __init__(self, num_classes=7, pretrained=True, dropout_cls=0.5):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────
        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove the classifier and the adaptive-pool head; keep feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Output: (B, 1536, 7, 7) for 224×224 input

        # ── Spatial Branch ────────────────────────────────────────
        self.drab = DRAB(in_channels=1536, mid_channels=512, dropout_p=0.4)

        # ── Frequency Branch ──────────────────────────────────────
        self.wtm = WaveletTransformModule(
            in_channels=1536, out_channels_per_band=128,
            target_size=(7, 7), dropout_p=0.3
        )

        # ── Fusion ────────────────────────────────────────────────
        self.cdf = CrossDomainFusion(channels=512)

        # ── Global Average Pool ───────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, 512)

        # ── Classifier ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_cls),
            nn.Linear(256, num_classes, bias=True),
        )

    def forward(self, x):
        # Backbone (shared)
        f_s = self.backbone(x)              # (B, 1536, 7, 7)  Eq. 1

        # Spatial branch
        y_pp = self.drab(f_s)               # (B, 512,  7, 7)  Eq. 2-7

        # Frequency branch
        f_w_prime = self.wtm(f_s)           # (B, 512,  7, 7)  Eq. 8-13

        # Fusion
        f_out = self.cdf(y_pp, f_w_prime)   # (B, 512,  7, 7)  Eq. 14-17

        # GAP → classifier
        h   = self.gap(f_out).flatten(1)    # (B, 512)          Eq. 18
        out = self.classifier(h)            # (B, num_classes)  Eq. 19-22
        return out


# ─────────────────────────────────────────────
# Ablation variants (Section 6)
# ─────────────────────────────────────────────
class Baseline(nn.Module):
    """EfficientNet-B3 + GAP + Linear classifier (no DRAB / WTM / CDF)."""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone   = nn.Sequential(*list(backbone.children())[:-2])
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1536, num_classes)

    def forward(self, x):
        f = self.gap(self.backbone(x)).flatten(1)
        return self.classifier(f)


class BaselinePlusDRAB(nn.Module):
    """Baseline + DRAB (no WTM / CDF)."""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone   = nn.Sequential(*list(backbone.children())[:-2])
        self.drab       = DRAB()
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.drab(f)
        f = self.gap(f).flatten(1)
        return self.classifier(f)


class BaselinePlusDRABWTM(nn.Module):
    """Baseline + DRAB + WTM, naive concatenation (no CDF)."""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        backbone = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone   = nn.Sequential(*list(backbone.children())[:-2])
        self.drab       = DRAB()
        self.wtm        = WaveletTransformModule()
        self.fuse_conv  = nn.Conv2d(1024, 512, 1, bias=False)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f_s    = self.backbone(x)
        y_pp   = self.drab(f_s)
        f_w    = self.wtm(f_s)
        fused  = self.fuse_conv(torch.cat([y_pp, f_w], dim=1))
        h      = self.gap(fused).flatten(1)
        return self.classifier(h)


if __name__ == "__main__":
    model = WAFENet(num_classes=7)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print("Output shape:", out.shape)          # (2, 7)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total:.1f} M")  # ≈ 14.3 M
