import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDilatedBlock(nn.Module):
    """
    A residual dilated block with 3 dilated convolutional layers

    """

    def __init__(self, in_channels=3, out_channels=1):
        """Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
        Returns:
            None
        """
        super().__init__()

        self.rdb = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )
        self.convID = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=2, padding=0
        )

    def forward(self, x):
        """Forward method.
        Args:
            x: input tensor
        Returns:
            out: output tensor
        """
        return self.rdb(x) + self.convID(x)


# Risidual block
class RisidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
        Returns:
            None
        """
        super().__init__()

        self.rdb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
        )
        self.convID = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """Forward method.
        Args:
            x: input tensor
        Returns:
            out: output tensor
        """
        return self.rdb(x) + self.convID(x)


# Double Convolution block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality§
        Returns:
            None
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.convID = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """Forward method.
        Args:
            x: input tensor
        Returns:
            out: output tensor
        """
        return self.double_conv(x) + self.convID(x)


# Upsampling block
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        """Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            dilation: dilation rate
        Returns:
            None
        """
        super().__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            ),  # 32x32 -> 64x64
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, input):
        """Forward method.
        Args:
            x: input tensor
            input: input tensor from the encoder
        Returns:
            out: output tensor
        """
        x = self.upconv(x)
        x = torch.cat([x, input], dim=1)
        return x


class PPM(nn.Module):
    def __init__(self, num_class, fc_dim, pool_scales=(1, 2, 3, 6)):
        """Constructor
        Args:
            num_class: number of classes
            fc_dim: number of filters in the last conv layer
            pool_scales: pooling scales
        Returns:
            None
        """
        super(PPM, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                )
            )

        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                fc_dim + len(pool_scales) * 512,
                512,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1),
        )

    def forward(self, x, seg_size=None):
        """Forward method.
        Args:
            x: input tensor
            seg_size: segmentation size
        Returns:
            x: output tensor
        """
        input_size = x.size()
        ppm_out = [x]

        for pool_scale in self.ppm:
            ppm_out.append(
                nn.functional.interpolate(
                    pool_scale(x),
                    (input_size[2], input_size[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if seg_size:  # is True during inference
            x = nn.functional.interpolate(
                x, size=seg_size, mode="bilinear", align_corners=False
            )
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# GC-DCNN model
class GCDCNN(nn.Module):
    def __init__(self, n_channels=3, num_classes=1):
        """Constructor
        Args:
            n_channels: input channel dimensionality
            n_classes: output channel dimensionality
        Returns:
            None
        """
        super(GCDCNN, self).__init__()

        # Encoder
        self.dc1 = DoubleConv(n_channels, 64)
        self.rdb1 = ResidualDilatedBlock(64, 128)
        self.rdb2 = ResidualDilatedBlock(128, 256)
        self.rdb3 = ResidualDilatedBlock(256, 512)

        # PPM
        self.ppm = PPM(1024, 512)

        # Decoder
        self.up1 = UpConv(1024, 256)
        self.rb1 = RisidualBlock(512, 256)
        self.up2 = UpConv(256, 128)
        self.rb2 = RisidualBlock(256, 128)
        self.up3 = UpConv(128, 64)
        self.rb3 = RisidualBlock(128, 64)

        # Output
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """Forward method.
        Args:
            x: input tensor
        Returns:
            logits: output tensor
        """
        x1 = self.dc1(x)
        x2 = self.rdb1(x1)
        x3 = self.rdb2(x2)
        x4 = self.rdb3(x3)

        x5 = self.ppm(x4)

        x = self.up1(x5, x3)
        x = self.rb1(x)
        x = self.up2(x, x2)
        x = self.rb2(x)
        x = self.up3(x, x1)
        x = self.rb3(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)
