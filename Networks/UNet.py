import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A double convolutional block that performs two consecutive sets of convolution,
    batch normalization, and ReLU operations.
    """

    def __init__(self, in_channels, out_channels):
        """
        constructor for DoubleConv class
        args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
        returns:
            None
        """
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        forward method for DoubleConv class
        args:
            x (tensor): input tensor
        returns:
            tensor: output tensor
        """
        return self.double_conv(x)


class UpConv(nn.Module):
    """
    An upsampling block using transposed convolutions.
    """

    def __init__(self, in_channels, out_channels):
        """
        constructor for UpConv class
        args:
            self: instance of class
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the transposed convolution.
        returns:
            None
        """
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        """
        forward method for UpConv class
        args:
            x (tensor): input tensor
        returns:
            tensor: output tensor
        """
        return self.up(x)


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.
    Attributes:
        dc1, dc2, dc3, dc4, dc5 (DoubleConv): The double convolution blocks.
        up1, up2, up3, up4 (UpConv): The upsampling blocks.
        dc6, dc7, dc8, dc9 (DoubleConv): The double convolution blocks.
        outc (nn.Conv2d): The final convolutional layer to output the segmentation map.
    """

    def __init__(self, n_channels=3, num_classes=1):
        """
        constructor for UNet class
        args:
            self: instance of class
            n_channels (int): Number of channels in the input image.
            num_classes (int): Number of classes for the segmentation task.
        returns:
            None
        """
        super(UNet, self).__init__()
        # Down
        self.dc1 = DoubleConv(n_channels, 64)
        self.dc2 = DoubleConv(64, 128)
        self.dc3 = DoubleConv(128, 256)
        self.dc4 = DoubleConv(256, 512)
        self.dc5 = DoubleConv(512, 1024)
        # Up
        self.up1 = UpConv(1024, 512)
        self.dc6 = DoubleConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.dc7 = DoubleConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.dc8 = DoubleConv(256, 128)
        self.up4 = UpConv(128, 64)
        self.dc9 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        """
        forward method for UNet class
        args:
            x (tensor): input tensor
        returns:
            tensor: output tensor
        """
        x1 = self.dc1(x)
        x2 = self.dc2(F.max_pool2d(x1, 2))
        x3 = self.dc3(F.max_pool2d(x2, 2))
        x4 = self.dc4(F.max_pool2d(x3, 2))
        x5 = self.dc5(F.max_pool2d(x4, 2))

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dc6(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dc7(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dc8(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dc9(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)
