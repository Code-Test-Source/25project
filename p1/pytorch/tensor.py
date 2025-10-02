import torch

# 创建示例tensor
x = torch.randn(2, 3, 4, 4)  # (batch, channel, height, width)

# 铺平操作
flattened = x.flatten(1)      # 从第1维开始铺平 -> (2, 48)
flattened = torch.flatten(x, 1)

# reshape
reshaped = x.reshape(2, -1)   # -1表示自动计算
reshaped = x.view(2, -1)

# 扩增维度
expanded = x.unsqueeze(0)     # 在0维增加维度 -> (1, 2, 3, 4, 4)
squeezed = x.squeeze()        # 删除所有长度为1的维度

# tensor拼接
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
concatenated = torch.cat([x1, x2], dim=1)  # -> (2, 6)
stacked = torch.stack([x1, x2], dim=0)     # -> (2, 2, 3)

# 非线性函数
activated = torch.relu(x)
activated = torch.sigmoid(x)
activated = torch.tanh(x)