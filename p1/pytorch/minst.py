#%%
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])

# 加载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输入32通道，输出64通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(128 * 2 * 2, 256)  # 展平后输入到全连接层
        self.fc2 = nn.Linear(256, 10)  # 10 个类别

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 第一层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv2(x))  # 第二层卷积 + ReLU
        x = F.max_pool2d(x, 2)     # 最大池化
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 2 * 2) # 展平
        x = F.relu(self.fc1(x))    # 全连接层 + ReLU
        x = self.fc2(x)            # 最后一层输出
        return x

# 创建模型实例
model = SimpleCNN()

# 3. 定义损失函数与优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. 模型训练
num_epochs = 5
model.train()  # 设置模型为训练模式

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失

        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

# 5. 模型测试
model.eval()  # 设置模型为评估模式
correct = 0
total = 0

with torch.no_grad():  # 关闭梯度计算
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 6. 可视化测试结果
dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predictions = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    axes[i].imshow(images[i][0], cmap='gray')
    axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
    axes[i].axis('off')
plt.show()
'''
IF i use conv1 and conv2 and layer [64,7,7],it produces the following result.
100.0%
100.0%
100.0%
100.0%
Epoch [1/5], Loss: 0.2321
Epoch [2/5], Loss: 0.0546
Epoch [3/5], Loss: 0.0382
Epoch [4/5], Loss: 0.0283
Epoch [5/5], Loss: 0.0237
Test Accuracy: 98.80%
'''
'''
The result of
self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)
self.fc1 = nn.Linear(128 * 2 * 2, 256)  # 展平后输入到全连接层
self.fc2 = nn.Linear(256, 10)  # 10 个类别
My optimized network is better than the unoptimized default network.
100%|██████████| 9.91M/9.91M [00:09<00:00, 1.03MB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 119kB/s]
100%|██████████| 1.65M/1.65M [00:05<00:00, 313kB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 3.34MB/s]
Epoch [1/5], Loss: 0.3200
Epoch [2/5], Loss: 0.0484
Epoch [3/5], Loss: 0.0319
Epoch [4/5], Loss: 0.0237
Epoch [5/5], Loss: 0.0172
Test Accuracy: 99.18%
'''
# There are many ways to play with it but I'll just stop there and continue to CIFAR10.