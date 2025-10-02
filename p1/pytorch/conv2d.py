import torch.nn as nn

# 定义2D卷积
conv = nn.Conv2d(
    in_channels=3,    # 输入通道数 (RGB图像为3)
    out_channels=64,  # 输出通道数 (卷积核数量)
    kernel_size=3,    # 卷积核大小 3x3
    stride=1,         # 步长
    padding=1,        # 填充
    bias=True         # 是否使用偏置
)

# 输出大小计算
def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1

# 示例: 输入224x224, 卷积后输出大小
output_h = conv_output_size(224, 3, 1, 1)  # 224


fc = nn.Linear(
    in_features=1024,  # 输入向量长度
    out_features=10    # 输出向量长度 (如10分类)
)


# 最大池化
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)


bn = nn.BatchNorm2d(
    num_features=64,  # 通道数
    eps=1e-05,        # 数值稳定性
    momentum=0.1      # 运行均值的动量
)