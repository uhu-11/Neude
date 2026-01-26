import jittor as jt
from jittor import nn
import numpy as np
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(out_channels)
        self.downsample = downsample

    def execute(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet20 网络结构
class ResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet20, self).__init__()
        self.in_channels = 16  # 初始输入通道数

        # 初始卷积层
        self.conv1 = nn.Conv(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(16)
        self.relu = nn.ReLU()

        # ResNet-20 的三个阶段（每个阶段由 BasicBlock 组成）
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # 全局平均池化层和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 如果需要下采样，则定义一个卷积层
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(out_channels)
            )

        # 构建每一层的块
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def execute(self, x):
        # 数据格式转换：B x C x H x W (从 NHWC -> NCHW)
        x = x.permute((0, 3, 1, 2)).float32()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 通过三个残差块阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全局平均池化
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        # 全连接层输出分类结果
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

# 创建一个 ResNet20 模型实例
def resnet20(num_classes=10):
    return ResNet20(BasicBlock, [3, 3, 3], num_classes=num_classes)