'''Loading CIFAR-100 dataset...
Using device: cuda
Training:   0%|          | 0/391 [00:00<?, ?it/s]Starting training...
Training: 100%|██████████| 391/391 [01:27<00:00,  4.46it/s, Loss=3.719, Acc=12.77%]
Testing: 100%|██████████| 79/79 [00:04<00:00, 19.38it/s, Loss=3.041, Acc=23.25%]
Epoch 1/10 - Time: 91.83s
Train Loss: 3.7194, Train Acc: 12.77%
Test Loss: 3.0410, Test Acc: 23.25%
--------------------------------------------------
New best model saved with accuracy: 23.25%
Training: 100%|██████████| 391/391 [01:34<00:00,  4.13it/s, Loss=2.979, Acc=25.20%]
Testing: 100%|██████████| 79/79 [00:03<00:00, 20.67it/s, Loss=2.636, Acc=32.13%]
Epoch 2/10 - Time: 98.48s
Train Loss: 2.9792, Train Acc: 25.20%
Test Loss: 2.6361, Test Acc: 32.13%
--------------------------------------------------
Training:   0%|          | 0/391 [00:00<?, ?it/s]New best model saved with accuracy: 32.13%
Training: 100%|██████████| 391/391 [01:32<00:00,  4.23it/s, Loss=2.648, Acc=31.91%]
Testing: 100%|██████████| 79/79 [00:03<00:00, 20.34it/s, Loss=2.492, Acc=35.35%]
Epoch 3/10 - Time: 96.30s
Train Loss: 2.6483, Train Acc: 31.91%
Test Loss: 2.4920, Test Acc: 35.35%
--------------------------------------------------
Training:   0%|          | 0/391 [00:00<?, ?it/s]New best model saved with accuracy: 35.35%
Training: 100%|██████████| 391/391 [01:38<00:00,  3.96it/s, Loss=2.440, Acc=36.50%]
Testing: 100%|██████████| 79/79 [00:03<00:00, 20.04it/s, Loss=2.195, Acc=42.09%]
Epoch 4/10 - Time: 102.57s
Train Loss: 2.4403, Train Acc: 36.50%
Test Loss: 2.1952, Test Acc: 42.09%
--------------------------------------------------
New best model saved with accuracy: 42.09%
Training: 100%|██████████| 391/391 [01:35<00:00,  4.09it/s, Loss=2.290, Acc=39.85%]
Testing: 100%|██████████| 79/79 [00:02<00:00, 28.84it/s, Loss=2.018, Acc=45.91%]
Epoch 5/10 - Time: 98.24s
Train Loss: 2.2903, Train Acc: 39.85%
Test Loss: 2.0177, Test Acc: 45.91%
--------------------------------------------------
New best model saved with accuracy: 45.91%
Training: 100%|██████████| 391/391 [02:44<00:00,  2.38it/s, Loss=2.160, Acc=42.55%]
Testing: 100%|██████████| 79/79 [00:03<00:00, 22.64it/s, Loss=1.961, Acc=47.28%]
Epoch 6/10 - Time: 168.10s
Train Loss: 2.1600, Train Acc: 42.55%
Test Loss: 1.9608, Test Acc: 47.28%
--------------------------------------------------
New best model saved with accuracy: 47.28%
Training: 100%|██████████| 391/391 [03:08<00:00,  2.08it/s, Loss=2.056, Acc=45.05%]
Testing: 100%|██████████| 79/79 [00:06<00:00, 11.82it/s, Loss=1.860, Acc=49.42%]
Epoch 7/10 - Time: 195.01s
Train Loss: 2.0556, Train Acc: 45.05%
Test Loss: 1.8596, Test Acc: 49.42%
--------------------------------------------------
Training:   0%|          | 0/391 [00:00<?, ?it/s]New best model saved with accuracy: 49.42%
Training: 100%|██████████| 391/391 [04:07<00:00,  1.58it/s, Loss=1.971, Acc=46.90%]
Testing: 100%|██████████| 79/79 [00:01<00:00, 50.41it/s, Loss=1.791, Acc=51.21%]
Epoch 8/10 - Time: 249.16s
Train Loss: 1.9708, Train Acc: 46.90%
Test Loss: 1.7915, Test Acc: 51.21%
--------------------------------------------------
Training:   0%|          | 0/391 [00:00<?, ?it/s]New best model saved with accuracy: 51.21%
Training: 100%|██████████| 391/391 [00:35<00:00, 10.96it/s, Loss=1.898, Acc=48.90%]
Testing: 100%|██████████| 79/79 [00:01<00:00, 51.28it/s, Loss=1.724, Acc=53.21%]
Epoch 9/10 - Time: 37.23s
Train Loss: 1.8979, Train Acc: 48.90%
Test Loss: 1.7244, Test Acc: 53.21%
--------------------------------------------------
Training:   0%|          | 0/391 [00:00<?, ?it/s]New best model saved with accuracy: 53.21%
Training: 100%|██████████| 391/391 [00:36<00:00, 10.68it/s, Loss=1.814, Acc=50.84%]
Testing: 100%|██████████| 79/79 [00:01<00:00, 48.82it/s, Loss=1.735, Acc=53.38%]
Epoch 10/10 - Time: 38.22s
Train Loss: 1.8135, Train Acc: 50.84%
Test Loss: 1.7346, Test Acc: 53.38%
--------------------------------------------------
New best model saved with accuracy: 53.38%
Training completed. Best accuracy: 53.38%
Loading best model for final evaluation...
Testing: 100%|██████████| 79/79 [00:01<00:00, 51.73it/s, Loss=1.735, Acc=53.38%]
Final Test Accuracy: 53.38%'''


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm
import time
import os

# 修复多进程问题
if __name__ == '__main__':
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),  # CIFAR-100的统计量
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # 加载CIFAR-100数据集
    print("Loading CIFAR-100 dataset...")
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # 数据加载器 - 在Windows上减少num_workers或设为0
    batch_size = 128
    num_workers = 0  # 在Windows上设为0避免多进程问题

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


    # 定义模型 - 修复过时的pretrained参数
    class CIFAR100Model(nn.Module):
        def __init__(self, num_classes=100):
            super().__init__()
            # 使用新的权重API
            self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

            # 修改第一层卷积，适应CIFAR-100的32x32输入
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

            # 移除原来的全连接层，替换为适合100类的层
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            return self.backbone(x)


    # 或者使用更简单的自定义CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=100):
            super().__init__()
            self.features = nn.Sequential(
                # 第一层
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),

                # 第二层
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),

                # 第三层
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.4),
            )

            self.classifier = nn.Sequential(
                nn.Linear(256 * 4 * 4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x


    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = CIFAR100Model(num_classes=100).to(device)
    # 或者使用简单CNN: model = SimpleCNN(num_classes=100).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    # 训练参数
    num_epochs = 10
    best_accuracy = 0


    # 训练函数
    def train_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.3f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        return running_loss / len(dataloader), 100. * correct / total


    # 测试函数
    def test_epoch(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Testing')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'Loss': f'{running_loss / (batch_idx + 1):.3f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        return running_loss / len(dataloader), 100. * correct / total


    # 开始训练
    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 测试
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print('-' * 50)

        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': test_acc,
                'loss': test_loss
            }, 'best_cifar100_model.pth')
            print(f'New best model saved with accuracy: {test_acc:.2f}%')

    print(f'Training completed. Best accuracy: {best_accuracy:.2f}%')

    # 加载最佳模型进行最终评估
    print("Loading best model for final evaluation...")
    checkpoint = torch.load('best_cifar100_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 最终测试
    final_loss, final_accuracy = test_epoch(model, test_loader, criterion, device)
    print(f'Final Test Accuracy: {final_accuracy:.2f}%')