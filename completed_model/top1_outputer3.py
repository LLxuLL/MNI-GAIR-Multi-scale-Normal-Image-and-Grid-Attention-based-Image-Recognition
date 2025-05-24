import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pandas.core.computation.expressions import evaluate
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# Deep convolutional network model (introducing residual connections and global average pooling)
class DeepResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(DeepResNet, self).__init__()
        self.fc = nn.Linear(256, num_classes)  # Modify the output dimensions
        # Enter the preprocessing layer
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Residual block group
        self.layer1 = self._make_residual_block(64, 64, stride=1)
        self.layer2 = self._make_residual_block(64, 128, stride=2)
        self.layer3 = self._make_residual_block(128, 256, stride=2)

        # Global Average Pooling Fully connected
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

        # Initialization parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_residual_block(self, in_channels, out_channels, stride):
        shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride, shortcut),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        x += residual
        return F.relu(x, inplace=True)


# Data augmentation and preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
