import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetTCN(nn.Module):
    def __init__(self, resnet_out=512, num_classes=2):
        super().__init__()
        base_model = resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(base_model.children())[:-1])  # Remove FC layer
        self.resnet_out = resnet_out

        self.tcn = nn.Sequential(
            nn.Conv1d(self.resnet_out, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape  # Batch, Time, Channel, H, W
        x = x.view(B * T, C, H, W)
        features = self.resnet(x).view(B, T, -1)  # (B, T, 512)

        features = features.permute(0, 2, 1)  # (B, 512, T)
        out = self.tcn(features)
        logits = self.classifier(out)
        return logits
