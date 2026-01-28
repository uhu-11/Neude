import jittor as jt
from jittor import nn
import numpy as np
class LeNet1(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet1, self).__init__()
        # 卷积层1: 输入1通道，输出4通道，卷积核大小5x5，步长为1，padding为0
        self.conv1 = nn.Conv(1, 4, kernel_size=5, stride=1, padding=0)
        # 平均池化层1: 2x2池化，步长为2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 卷积层2: 输入4通道，输出8通道，卷积核大小5x5，步长为1，padding为0
        self.conv2 = nn.Conv(4, 8, kernel_size=5, stride=1, padding=0)
        # 平均池化层2: 2x2池化，步长为2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层: 输入大小为8 * 4 * 4（因为经过两次池化后，特征图的大小变为4x4），输出10个类别
        self.fc = nn.Linear(8 * 4 * 4, num_classes)


    def execute(self, x):
        x = x.permute((0, 3, 1, 2)).float32()

        x = nn.relu(self.conv1(x))
        x = self.pool1(x)

        x = nn.relu(self.conv2(x))
        x = self.pool2(x)

        # 展平数据，准备传入全连接层
        x = x.view(x.shape[0], -1)
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
        # 确保推理时不计算梯度
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

def lenet1(num_classes=10):
    return LeNet1(num_classes=num_classes)