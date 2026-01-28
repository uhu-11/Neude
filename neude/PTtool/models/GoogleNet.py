import jittor as jt
from jittor import nn
import numpy as np

# 定义 Inception 模块（GoogleNet 的核心构建块）
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels_1x1, out_channels_3x3, out_channels_5x5, out_channels_pool):
        super(Inception, self).__init__()
        # 1x1 卷积
        self.conv1x1 = nn.Conv2d(in_channels, out_channels_1x1, kernel_size=1, stride=1, padding=0, bias=False)

        # 1x1 -> 3x3 卷积
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels_3x3, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3_2 = nn.Conv2d(out_channels_3x3, out_channels_3x3, kernel_size=3, stride=1, padding=1, bias=False)

        # 1x1 -> 5x5 卷积
        self.conv5x5_1 = nn.Conv2d(in_channels, out_channels_5x5, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv5x5_2 = nn.Conv2d(out_channels_5x5, out_channels_5x5, kernel_size=5, stride=1, padding=2, bias=False)

        # 3x3 最大池化 -> 1x1 卷积
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool_conv = nn.Conv2d(in_channels, out_channels_pool, kernel_size=1, stride=1, padding=0, bias=False)

    def execute(self, x):
        # Inception 模块的不同路径
        path1 = self.conv1x1(x)

        path2 = self.conv3x3_1(x)
        path2 = self.conv3x3_2(path2)

        path3 = self.conv5x5_1(x)
        path3 = self.conv5x5_2(path3)

        path4 = self.pool(x)
        path4 = self.pool_conv(path4)

        # 拼接所有路径
        out = jt.contrib.concat([path1, path2, path3, path4], dim=1)
        return out


# 定义 GoogleNet 网络结构
class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()

        # 最大池化
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Inception 模块
        self.inception1 = Inception(64, 32, 32, 32, 32)
        self.inception2 = Inception(128, 64, 64, 64, 64)
        self.inception3 = Inception(256, 128, 128, 128, 128)

        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc = nn.Linear(512, num_classes)  # 输出通道为 512（来自 Inception 模块的输出）

    def execute(self, x):
        x = x.permute((0, 3, 1, 2)).float32()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 通过 Inception 模块
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)

        # 全局平均池化
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # 扁平化

        # 全连接层
        x = self.fc(x)

        return x

    def predict(self, x, apply_softmax=False):
        """
        自定义 predict 方法，用于推理阶段
        :param x: 输入数据，支持 Jittor Tensor 或 NumPy 数组
        :param apply_softmax: 是否对输出应用 softmax 激活
        :return: 预测结果（Jittor Tensor 或 NumPy 数组）
        """
        # 禁用梯度计算
        with jt.no_grad():
            # 如果输入是 NumPy 数组或列表，转换为 Jittor Tensor
            if isinstance(x, (list, tuple)):
                x = jt.array(x)
            elif isinstance(x, np.ndarray):
                x = jt.array(x)

            # 执行前向传播
            logits = self.execute(x)

            # 可选：对输出应用 softmax
            if apply_softmax:
                predictions = jt.nn.softmax(logits, dim=-1)
            else:
                predictions = logits

        return predictions


# 创建一个 GoogleNet 模型实例
def googlenet(num_classes=10):
    return GoogleNet(num_classes=num_classes)

