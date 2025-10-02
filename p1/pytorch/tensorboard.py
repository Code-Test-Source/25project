from torch.utils.tensorboard import SummaryWriter
import datetime

# 创建writer
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f'runs/experiment_{timestamp}')


# 训练过程中记录
def train_with_logging(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)

        # 记录到tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # 记录模型权重分布
        for name, param in model.named_parameters():
            writer.add_histogram(f'weights/{name}', param, epoch)
            writer.add_histogram(f'grads/{name}', param.grad, epoch)

    writer.close()


# 验证函数
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return total_loss / len(dataloader), correct / total