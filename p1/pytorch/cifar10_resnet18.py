'''Using device: cuda
Starting Training...
Epoch 1/10: 100%|██████████| 391/391 [00:45<00:00,  8.56it/s, Loss=1.151, Acc=38.03%]
Epoch 1 completed in 45.67s, Loss: 1.151, Train Acc: 38.03%, Test Acc: 48.40%
Epoch 2/10: 100%|██████████| 391/391 [00:44<00:00,  8.78it/s, Loss=0.818, Acc=53.34%]
Epoch 2 completed in 44.55s, Loss: 0.818, Train Acc: 53.34%, Test Acc: 52.29%
Epoch 3/10: 100%|██████████| 391/391 [00:47<00:00,  8.25it/s, Loss=0.677, Acc=59.97%]
Epoch 3 completed in 47.39s, Loss: 0.677, Train Acc: 59.97%, Test Acc: 64.65%
Epoch 4/10: 100%|██████████| 391/391 [00:45<00:00,  8.51it/s, Loss=0.586, Acc=64.51%]
Epoch 4 completed in 45.95s, Loss: 0.586, Train Acc: 64.51%, Test Acc: 72.49%
Epoch 5/10: 100%|██████████| 391/391 [00:45<00:00,  8.65it/s, Loss=0.516, Acc=68.35%]
Epoch 6/10:   0%|          | 0/391 [00:00<?, ?it/s]Epoch 5 completed in 45.22s, Loss: 0.516, Train Acc: 68.35%, Test Acc: 71.60%
Epoch 6/10: 100%|██████████| 391/391 [00:46<00:00,  8.49it/s, Loss=0.457, Acc=71.54%]
Epoch 6 completed in 46.07s, Loss: 0.457, Train Acc: 71.54%, Test Acc: 77.19%
Epoch 7/10: 100%|██████████| 391/391 [00:45<00:00,  8.52it/s, Loss=0.404, Acc=74.34%]
Epoch 7 completed in 45.89s, Loss: 0.404, Train Acc: 74.34%, Test Acc: 79.25%
Epoch 8/10: 100%|██████████| 391/391 [00:45<00:00,  8.54it/s, Loss=0.361, Acc=76.74%]
Epoch 8 completed in 45.81s, Loss: 0.361, Train Acc: 76.74%, Test Acc: 82.81%
Epoch 9/10: 100%|██████████| 391/391 [00:46<00:00,  8.49it/s, Loss=0.330, Acc=78.40%]
Epoch 9 completed in 46.04s, Loss: 0.330, Train Acc: 78.40%, Test Acc: 84.35%
Epoch 10/10: 100%|██████████| 391/391 [00:46<00:00,  8.38it/s, Loss=0.311, Acc=79.59%]
Epoch 10 completed in 46.67s, Loss: 0.311, Train Acc: 79.59%, Test Acc: 85.20%
Finished Training, Best Accuracy: 85.20%
Testing:   0%|          | 0/79 [00:00<?, ?it/s]Starting Final Testing...
Testing: 100%|██████████| 79/79 [00:02<00:00, 31.82it/s]

Overall Accuracy: 85.20%

Per-class Accuracy:
plane     : 86.00% (860/1000)
car       : 96.00% (960/1000)
bird      : 76.20% (762/1000)
cat       : 66.10% (661/1000)
deer      : 83.30% (833/1000)
dog       : 79.20% (792/1000)
frog      : 91.20% (912/1000)
horse     : 88.40% (884/1000)
ship      : 94.20% (942/1000)
truck     : 91.40% (914/1000)'''


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import platform
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# 根据操作系统设置num_workers
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 4

# 增强的数据增强和归一化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    transforms.RandomErasing(p=0.2)  # 新增随机擦除
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 优化后的超参数设置
batch_size = 128
learning_rate = 0.1
epochs = 10  # 增加训练轮数
momentum = 0.9
weight_decay = 1e-4  # 调整权重衰减


# 改进的网络结构 - 使用残差连接
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ImprovedNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2], num_classes=10):
        super(ImprovedNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 残差块
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)

        # 全局平均池化替代全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


# 焦点损失函数处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 主程序入口
if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=False,
        transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=False,
        transform=test_transform
    )

    # 数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 初始化网络、损失函数和优化器
    net = ImprovedNet().to(device)

    # 使用焦点损失处理类别不平衡
    criterion = FocalLoss(alpha=1, gamma=2)
    # 或者使用带权重的交叉熵
    # class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay, nesterov=True)

    # 改进的学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    # 或者使用OneCycleLR
    # scheduler = OneCycleLR(optimizer, max_lr=0.1, epochs=epochs, steps_per_epoch=len(trainloader))

    # 训练网络
    print("Starting Training...")
    best_accuracy = 0

    for epoch in range(epochs):
        start_time = time.time()
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示进度条
        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            # 统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.3f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

            # 更新学习率 (如果是OneCycleLR)
            # scheduler.step()

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - start_time

        # 每个epoch结束后在测试集上验证
        net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100. * test_correct / test_total

        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s, '
              f'Loss: {running_loss / len(trainloader):.3f}, '
              f'Train Acc: {100. * correct / total:.2f}%, '
              f'Test Acc: {test_accuracy:.2f}%')

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            PATH = './cifar_improved_net_best.pth'
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'epoch': epoch
            }, PATH)

    print(f'Finished Training, Best Accuracy: {best_accuracy:.2f}%')

    # 加载最佳模型进行最终测试
    checkpoint = torch.load('./cifar_improved_net_best.pth')
    net.load_state_dict(checkpoint['model_state_dict'])

    # 最终测试
    print("Starting Final Testing...")
    net.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in tqdm(testloader, desc='Testing'):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            # 统计总体准确率
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # 统计每个类别的准确率
            for label, prediction in zip(labels, predictions):
                class_name = classes[label]
                if label == prediction:
                    correct_pred[class_name] += 1
                total_pred[class_name] += 1

    # 打印总体准确率
    overall_accuracy = 100 * total_correct / total_samples
    print(f'\nOverall Accuracy: {overall_accuracy:.2f}%')

    # 打印每个类别的准确率
    print('\nPer-class Accuracy:')
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'{classname:10s}: {accuracy:.2f}% ({correct_count}/{total_pred[classname]})')