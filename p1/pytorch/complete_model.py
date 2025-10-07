import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==================== 1. 数据准备 ====================
class DataProcessor:
    """数据处理器，生成模拟数据用于演示"""

    def __init__(self, n_samples=1000, img_size=32, n_classes=10):
        self.n_samples = n_samples
        self.img_size = img_size
        self.n_classes = n_classes

    def generate_data(self):
        """生成模拟图像数据"""
        # 生成随机图像数据 [batch, channels, height, width]
        X = np.random.randn(self.n_samples, 3, self.img_size, self.img_size).astype(np.float32)

        # 生成标签（多分类）
        y = np.random.randint(0, self.n_classes, self.n_samples)

        # 添加一些模式使数据可学习
        for i in range(self.n_samples):
            class_id = y[i]
            # 为每个类别添加特定的模式
            X[i, 0, class_id:class_id + 3, class_id:class_id + 3] = 1.0  # 红色通道
            X[i, 1, class_id:class_id + 2, class_id:class_id + 2] = 0.5  # 绿色通道

        return X, y

    def prepare_dataloaders(self, batch_size=32, test_size=0.2):
        """准备训练和测试数据加载器"""
        X, y = self.generate_data()

        # 转换为PyTorch张量
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y).long()  # Use long() instead of default

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor, y_tensor, test_size=test_size, random_state=42, stratify=y
        )

        # 创建数据集
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        return train_loader, test_loader


# ==================== 2. 模型定义 ====================
class FeatureExtractor(nn.Module):
    """特征提取器 - 卷积部分"""

    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()

        self.conv_layers = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # 第二层卷积
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # 第三层卷积
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        return self.conv_layers(x)


class Classifier(nn.Module):
    """分类器 - 全连接部分"""

    def __init__(self, input_size, num_classes, hidden_size=512):
        super(Classifier, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.fc_layers(x)


class CompleteCNN(nn.Module):
    """完整的CNN模型，组合特征提取器和分类器"""

    def __init__(self, in_channels=3, img_size=32, num_classes=10):
        super(CompleteCNN, self).__init__()

        self.features = FeatureExtractor(in_channels)
        self.num_features = 128 * (img_size // 8) * (img_size // 8)  # 经过3次池化

        self.classifier = Classifier(self.num_features, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取
        x = self.features(x)

        # 铺平
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)

        return x

    def freeze_features(self, freeze=True):
        """冻结特征提取器参数"""
        for param in self.features.parameters():
            param.requires_grad = not freeze

        print(f"Feature extractor {'frozen' if freeze else 'unfrozen'}")

    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True

        print("All parameters unfrozen")


# ==================== 3. 训练器类 ====================
class ModelTrainer:
    """模型训练器"""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.long()  # Add this line - Convert to long
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader, criterion):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.long()  # Add this line here too
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=25):
        """完整训练过程"""
        print("Starting training...")

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)

            # 学习率调度
            if scheduler:
                scheduler.step()

            # 记录指标
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f'Epoch: {epoch + 1:02d}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        print("Training completed!")

    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# ==================== 4. 模型工具函数 ====================
class ModelUtils:
    """模型工具类"""

    @staticmethod
    def save_model(model, path, optimizer=None, scheduler=None, epoch=None, loss=None):
        """保存模型"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @staticmethod
    def load_model(model, path, optimizer=None, scheduler=None):
        """加载模型"""
        if not os.path.exists(path):
            print(f"Model file {path} not found!")
            return None

        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Model loaded from {path}")
        print(f"Training stopped at epoch: {checkpoint.get('epoch', 'N/A')}, "
              f"Loss: {checkpoint.get('loss', 'N/A'):.4f}")

        return checkpoint.get('epoch', 0)

    @staticmethod
    def load_partial_weights(model, pretrained_path, strict=True):
        """部分加载预训练权重"""
        if not os.path.exists(pretrained_path):
            print(f"Pretrained model {pretrained_path} not found!")
            return model

        pretrained_dict = torch.load(pretrained_path, map_location='cpu')

        if 'model_state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_state_dict']

        model_dict = model.state_dict()

        # 过滤不匹配的键
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and model_dict[k].shape == v.shape}

        # 更新模型权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=strict)

        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
        return model

    @staticmethod
    def predict(model, dataloader, device):
        """使用模型进行预测"""
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                probabilities = F.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)


# ==================== 5. 主函数 ====================
def main():
    # 参数设置
    config = {
        'batch_size': 64,
        'epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'img_size': 32,
        'num_classes': 10
    }

    # 1. 准备数据
    print("Preparing data...")
    data_processor = DataProcessor(n_samples=2000, img_size=config['img_size'],
                                   n_classes=config['num_classes'])
    train_loader, test_loader = data_processor.prepare_dataloaders(
        batch_size=config['batch_size']
    )

    # 2. 创建模型
    print("Creating model...")
    model = CompleteCNN(
        in_channels=3,
        img_size=config['img_size'],
        num_classes=config['num_classes']
    ).to(device)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 4. 创建训练器
    trainer = ModelTrainer(model, device)

    # 5. 训练模型
    trainer.train(
        train_loader, test_loader, criterion, optimizer,
        scheduler=scheduler, epochs=config['epochs']
    )

    # 6. 绘制训练历史
    trainer.plot_training_history()

    # 7. 在测试集上评估
    print("\nEvaluating on test set...")
    predictions, targets, probabilities = ModelUtils.predict(model, test_loader, device)
    test_accuracy = accuracy_score(targets, predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(targets, predictions))

    # 8. 保存模型
    print("\nSaving model...")
    os.makedirs('models', exist_ok=True)
    ModelUtils.save_model(
        model,
        'models/complete_cnn_model.pth',
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config['epochs'],
        loss=trainer.val_losses[-1]
    )

    # 9. 演示模型加载
    print("\nDemonstrating model loading...")
    new_model = CompleteCNN(
        in_channels=3,
        img_size=config['img_size'],
        num_classes=config['num_classes']
    ).to(device)

    # 加载保存的权重
    ModelUtils.load_model(new_model, 'models/complete_cnn_model.pth')

    # 验证加载的模型性能
    new_predictions, _, _ = ModelUtils.predict(new_model, test_loader, device)
    new_accuracy = accuracy_score(targets, new_predictions)
    print(f"Loaded model Test Accuracy: {new_accuracy:.4f}")

    # 10. 演示参数冻结
    print("\nDemonstrating parameter freezing...")
    model.freeze_features(freeze=True)

    # 检查冻结后的可训练参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params:,}")

    # 解冻所有参数
    model.unfreeze_all()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after unfreezing: {trainable_params:,}")


if __name__ == "__main__":
    main()