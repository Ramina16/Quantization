import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub


VGG_13_ARCH = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG_13(nn.Module):
    """
    
    """
    def __init__(self, num_classes: int = 10) -> None:
        """
        
        """
        super(VGG_13, self).__init__()

        self.num_classes = num_classes
        layers = []

        in_channels = 3
        for x in VGG_13_ARCH:
            if x != 'M':
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                in_channels = x 
            else:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._create_classifier()

    def _create_classifier(self):
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class QVGG_13(nn.Module):
    """
    
    """
    def __init__(self, num_classes: int = 10) -> None:
        """
        
        """
        super(QVGG_13, self).__init__()

        self.num_classes = num_classes
        self.layers = []

        in_channels = 3
        for x in VGG_13_ARCH:
            if x != 'M':
                self.layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                self.layers.append(nn.BatchNorm2d(x))
                self.layers.append(nn.ReLU(inplace=True))
                in_channels = x 
            else:
                self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._create_classifier()

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _create_classifier(self):
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Linear(512, self.num_classes)
        )

    def fuse_model(self):
        self.eval()
        fused_modules = []
        for i in range(len(self.layers) - 1):
            if isinstance(self.layers[i], nn.Conv2d) and isinstance(self.layers[i + 1], nn.BatchNorm2d):
                if i + 2 < len(self.layers) and isinstance(self.layers[i + 2], nn.ReLU):
                    fused_modules.append([str(i), str(i + 1), str(i + 2)])
                else:
                    fused_modules.append([str(i), str(i + 1)])
    
        # Apply fusion
        torch.ao.quantization.fuse_modules(self.features, fused_modules, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        """
        x = self.quant(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)

        return x
