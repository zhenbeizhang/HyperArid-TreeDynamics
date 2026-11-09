import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks:
    Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → Dropout

    This module is used as the basic feature extraction unit
    throughout the encoder and decoder.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_rate=0.1):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Encoder downsampling block:
    MaxPool → DoubleConv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Decoder upsampling block:
    Upsample → Feature concatenation → DoubleConv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2,
            diff_y // 2, diff_y - diff_y // 2
        ])

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class MorphoAttention(nn.Module):
    """
    Morphological spatial attention module.

    This module enhances structural features by approximating
    morphological dilation and erosion operations through
    convolutional filters. The resulting boundary-enhanced
    attention map emphasizes tree crown boundaries.
    """

    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()

        self.channel_pool = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(inplace=True)
        )

        self.dilation_conv = nn.Sequential(
            nn.Conv2d(in_channels // reduction_ratio, 1, 3, padding=2, dilation=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.erosion_conv = nn.Sequential(
            nn.Conv2d(in_channels // reduction_ratio, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Conv2d(2, 1, 1)

    def forward(self, x):

        compressed = self.channel_pool(x)

        dilation_map = self.dilation_conv(compressed)

        erosion_map = self.erosion_conv(compressed)

        boundary = dilation_map - erosion_map

        boundary = self.edge_conv(boundary)

        fused = self.fusion(torch.cat([boundary, dilation_map], dim=1))

        spatial_att = torch.sigmoid(fused)

        return x * spatial_att


class DirectionCentralityModule(nn.Module):
    """
    Directional centrality module.

    This module approximates local directional consistency by
    measuring the variance of responses across multiple
    orientation-specific filters. High directional variance
    indicates boundary structures.
    """

    def __init__(self, in_channels, k=8):
        super().__init__()

        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.GroupNorm(4, in_channels // 4),
            nn.ReLU()
        )

        self.direction_conv = nn.Conv2d(
            in_channels // 4,
            8,
            kernel_size=3,
            padding=1,
            groups=8
        )

        nn.init.constant_(self.direction_conv.weight, 0)

        for i in range(8):

            self.direction_conv.weight.data[i, 0] = torch.tensor(
                [
                    [np.cos(i * np.pi / 4), 0, -np.cos(i * np.pi / 4)],
                    [0, 0, 0],
                    [-np.cos(i * np.pi / 4), 0, np.cos(i * np.pi / 4)]
                ]
            )

    def forward(self, x):

        x = self.conv(x)

        direction_responses = self.direction_conv(x)

        mean_response = direction_responses.mean(dim=1, keepdim=True)

        variance = torch.mean(
            (direction_responses - mean_response) ** 2,
            dim=1,
            keepdim=True
        )

        return torch.sigmoid(variance)


class AttentionGate(nn.Module):
    """
    Gated attention mechanism used in the decoder.

    The gate signal from the decoder modulates encoder
    features to suppress irrelevant responses before
    skip-connection fusion.
    """

    def __init__(self, gate_channels, input_channels, inter_channels=None):
        super().__init__()

        if inter_channels is None:
            inter_channels = input_channels // 2

        self.gate_conv = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, inter_channels, 1),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, input):

        gate = F.interpolate(gate, size=input.shape[2:], mode="bilinear", align_corners=True)

        g = self.gate_conv(gate)

        x = self.input_conv(input)

        psi = self.relu(g + x)

        psi = self.psi(psi)

        return input * psi


class UNet(nn.Module):
    """
    Improved U-Net architecture for individual tree crown detection.

    The model integrates three additional components:
    1. Morphological spatial attention (MorphoAttention)
    2. Directional centrality module (DCM)
    3. Gated skip connections (AttentionGate)
    """

    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.dcm1 = DirectionCentralityModule(64)
        self.dcm2 = DirectionCentralityModule(128)
        self.dcm3 = DirectionCentralityModule(256)

        self.MorphoAttention1 = MorphoAttention(64)
        self.MorphoAttention2 = MorphoAttention(128)
        self.MorphoAttention3 = MorphoAttention(256)
        self.MorphoAttention4 = MorphoAttention(512)

        self.gate1 = AttentionGate(512, 512)
        self.gate2 = AttentionGate(256, 256)
        self.gate3 = AttentionGate(128, 128)
        self.gate4 = AttentionGate(64, 64)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):

        x1 = self.inc(x)
        x1_att = self.MorphoAttention1(x1) * self.dcm1(x1)

        x2 = self.down1(x1_att)
        x2_att = self.MorphoAttention2(x2) * self.dcm2(x2)

        x3 = self.down2(x2_att)
        x3_att = self.MorphoAttention3(x3) * self.dcm3(x3)

        x4 = self.down3(x3_att)
        x4_att = self.MorphoAttention4(x4)

        x5 = self.down4(x4_att)

        x4_g = self.gate1(x5, x4_att)
        x = self.up1(x5, x4_g)

        x3_g = self.gate2(x, x3_att)
        x = self.up2(x, x3_g)

        x2_g = self.gate3(x, x2_att)
        x = self.up3(x, x2_g)

        x1_g = self.gate4(x, x1_att)
        x = self.up4(x, x1_g)

        logits = self.outc(x)

        return torch.sigmoid(logits)


def unet(n_channels, n_classes=1, bilinear=True):

    return UNet(n_channels, n_classes, bilinear)