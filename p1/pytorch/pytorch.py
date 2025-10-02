import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # 假设输入32x32，计算后为8x8
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 铺平
        x = self.classifier(x)
        return x


class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.backbone = SimpleCNN()
        self.head = nn.Linear(10, 5)  # 新的分类头

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output