import jittor as jt
from jittor import nn
import numpy as np
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv(1, 6, kernel_size=5, stride=1, padding=2)  # 输入通道1，输出通道6，卷积核5x5
        self.pool1 = nn.MaxPool2d(2, 2)  # 2x2 最大池化
        self.conv2 = nn.Conv(6, 16, kernel_size=5, stride=1, padding=0)  # 输入通道6，输出通道16，卷积核5x5
        self.pool2 = nn.MaxPool2d(2, 2)  # 2x2 最大池化

        # 调整fc1的输入大小，根据conv2的输出计算
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16通道, 5x5的特征图
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # 10个类别输出


    def execute(self, x):
            x = x.permute((0, 3, 1, 2)).float32()
            x = self.pool1(nn.relu(self.conv1(x)))
            x = self.pool2(nn.relu(self.conv2(x)))
            x = x.view(x.shape[0], -1)
            x = nn.relu(self.fc1(x))
            x = nn.relu(self.fc2(x))
            x = self.fc3(x)
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

def lenet5(num_classes=10):
    return LeNet5(num_classes=num_classes)