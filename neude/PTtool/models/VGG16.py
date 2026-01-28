import jittor as jt
from jittor import nn
import numpy as np

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()

        # 定义 VGG16 的卷积部分（8 层卷积）
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv4
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 定义全连接部分
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.resize = nn.Resize((128, 128))

    def execute(self, x):
        x = x.permute((0, 3, 1, 2)).float32()
        x = self.resize(x)
        x = self.features(x)
        x = x.view(x.shape[0], -1)  # 展平特征
        x = self.classifier(x)
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

def vgg16(num_classes=10):
    return VGG16(num_classes=num_classes)