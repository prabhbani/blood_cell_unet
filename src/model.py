"""UNet implemented from scratch in PyTorch."""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Classic 4-level UNet (encoder, bottleneck, decoder with skip connections)."""

    def __init__(self, in_ch=3, n_classes=4, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1, self.enc2 = DoubleConv(in_ch, c[0]), DoubleConv(c[0], c[1])
        self.enc3, self.enc4 = DoubleConv(c[1], c[2]), DoubleConv(c[2], c[3])
        self.bott = DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.up4, self.dec4 = nn.ConvTranspose2d(c[4], c[3], 2, 2), DoubleConv(c[4], c[3])
        self.up3, self.dec3 = nn.ConvTranspose2d(c[3], c[2], 2, 2), DoubleConv(c[3], c[2])
        self.up2, self.dec2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), DoubleConv(c[2], c[1])
        self.up1, self.dec1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bott(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


if __name__ == "__main__":
    m = UNet()
    print(sum(p.numel() for p in m.parameters()) / 1e6, "M params")
    print(m(torch.randn(2, 3, 240, 320)).shape)
