import torch
from torch import nn


class MFM(nn.Module):
    def __init__(
        self,
        in_c: int | None,
        out_c: int,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        mode: str = "conv",
    ):
        super().__init__()
        if mode == "conv":
            if in_c is None:
                raise ValueError("in_c must be set for conv MFM")
            self.filter = nn.Conv2d(in_c, out_c * 2, kernel_size, stride, padding)
        elif mode == "linear":
            self.filter = nn.LazyLinear(out_c * 2)
        else:
            raise ValueError("mode must be 'conv' or 'linear'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        return torch.maximum(x1, x2)


class LightCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            MFM(64, 32),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            MFM(64, 32),
            nn.BatchNorm2d(32),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 96, 3, 1, 1),
            MFM(96, 48),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.BatchNorm2d(48),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 96, 1),
            MFM(96, 48),
            nn.BatchNorm2d(48),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(48, 128, 3, 1, 1),
            MFM(128, 64),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            MFM(128, 64),
            nn.BatchNorm2d(64),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            MFM(64, 32),
            nn.BatchNorm2d(32),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(32, 64, 1),
            MFM(64, 32),
            nn.BatchNorm2d(32),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            MFM(64, 32),
            nn.MaxPool2d(2, 2, ceil_mode=True),
        )

        self.fix_pool = nn.AdaptiveAvgPool2d((53, 37))

        self.fc29 = nn.Linear(32 * 53 * 37, 160)
        self.mfm30 = MFM(None, 80, mode="linear")
        self.bn31 = nn.BatchNorm1d(80)
        self.dropout = nn.Dropout(0.75)
        self.fc32 = nn.Linear(80, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        x = self.fix_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc29(x)
        x = self.mfm30(x)
        x = self.bn31(x)
        x = self.dropout(x)
        return self.fc32(x)
