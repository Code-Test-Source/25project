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



import torch.optim as optim
import torch

# 模型实例化
model = SimpleCNN(num_classes=10)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练循环
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()  # 梯度清零
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    return running_loss / len(dataloader)


# 预测
def predict(model, data):
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(data)
        probabilities = torch.softmax(output, dim=1)
        predictions = torch.argmax(output, dim=1)
    return predictions, probabilities

# 保存整个模型
torch.save(model, 'model.pth')

# 保存模型参数
torch.save(model.state_dict(), 'model_params.pth')

# 加载整个模型
model = torch.load('model.pth')

# 加载模型参数
model = SimpleCNN()
model.load_state_dict(torch.load('model_params.pth'))

# 部分加载参数
pretrained_dict = torch.load('pretrained.pth')
model_dict = model.state_dict()

# 过滤不匹配的键
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 冻结参数
for name, param in model.named_parameters():
    if 'conv' in name:  # 冻结所有卷积层
        param.requires_grad = False

# 只训练需要梯度的参数
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)