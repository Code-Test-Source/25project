'''
Using device: cuda
100%|██████████| 170M/170M [00:14<00:00, 12.0MB/s]
Epoch 1/10:   0%|          | 0/391 [00:00<?, ?it/s]Starting Training...
Epoch 1/10: 100%|██████████| 391/391 [00:15<00:00, 24.98it/s, Loss=1.553, Acc=42.92%]
Epoch 1 completed in 15.65s, Loss: 1.553, Accuracy: 42.92%
Epoch 2/10: 100%|██████████| 391/391 [00:15<00:00, 25.55it/s, Loss=1.068, Acc=62.23%]
Epoch 2 completed in 15.30s, Loss: 1.068, Accuracy: 62.23%
Epoch 3/10: 100%|██████████| 391/391 [00:14<00:00, 26.49it/s, Loss=0.870, Acc=69.61%]
Epoch 3 completed in 14.76s, Loss: 0.870, Accuracy: 69.61%
Epoch 4/10: 100%|██████████| 391/391 [00:14<00:00, 26.24it/s, Loss=0.766, Acc=73.38%]
Epoch 4 completed in 14.90s, Loss: 0.766, Accuracy: 73.38%
Epoch 5/10: 100%|██████████| 391/391 [00:15<00:00, 25.31it/s, Loss=0.700, Acc=76.02%]
Epoch 5 completed in 15.45s, Loss: 0.700, Accuracy: 76.02%
Epoch 6/10: 100%|██████████| 391/391 [00:14<00:00, 26.37it/s, Loss=0.652, Acc=77.65%]
Epoch 6 completed in 14.83s, Loss: 0.652, Accuracy: 77.65%
Epoch 7/10: 100%|██████████| 391/391 [00:14<00:00, 26.36it/s, Loss=0.630, Acc=78.32%]
Epoch 7 completed in 14.84s, Loss: 0.630, Accuracy: 78.32%
Epoch 8/10: 100%|██████████| 391/391 [00:14<00:00, 26.46it/s, Loss=0.599, Acc=79.44%]
Epoch 8 completed in 14.77s, Loss: 0.599, Accuracy: 79.44%
Epoch 9/10: 100%|██████████| 391/391 [00:14<00:00, 26.61it/s, Loss=0.577, Acc=80.53%]
Epoch 9 completed in 14.69s, Loss: 0.577, Accuracy: 80.53%
Epoch 10/10: 100%|██████████| 391/391 [00:14<00:00, 26.62it/s, Loss=0.571, Acc=80.58%]
Epoch 10 completed in 14.69s, Loss: 0.571, Accuracy: 80.58%
Finished Training
Model saved to ./cifar_improved_net.pth
Starting Testing...
Testing: 100%|██████████| 79/79 [00:01<00:00, 43.48it/s]

Overall Accuracy: 72.76%

Per-class Accuracy:
plane     : 73.40% (734/1000)
car       : 96.10% (961/1000)
bird      : 77.30% (773/1000)
cat       : 25.70% (257/1000)
deer      : 53.80% (538/1000)
dog       : 86.20% (862/1000)
frog      : 84.00% (840/1000)
horse     : 74.00% (740/1000)
ship      : 81.60% (816/1000)
truck     : 75.50% (755/1000)
'''



import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time
import platform

# 根据操作系统设置num_workers
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 4

# 数据增强和归一化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 超参数设置
batch_size = 128
learning_rate = 0.1
epochs = 10
momentum = 0.9
weight_decay = 5e-4


# 改进的网络结构
class ImprovedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 第一层卷积块
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第二层卷积块
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第三层卷积块
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 主程序入口
if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        num_workers=num_workers,  # 使用动态设置的num_workers
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # 使用动态设置的num_workers
        pin_memory=True
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 初始化网络、损失函数和优化器
    net = ImprovedNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=momentum, weight_decay=weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # 训练网络
    print("Starting Training...")
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

        # 更新学习率
        scheduler.step()

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s, '
              f'Loss: {running_loss / len(trainloader):.3f}, '
              f'Accuracy: {100. * correct / total:.2f}%')

    print('Finished Training')

    # 保存模型
    PATH = './cifar_improved_net.pth'
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, PATH)
    print(f'Model saved to {PATH}')

    # 测试模型
    print("Starting Testing...")
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